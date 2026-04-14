import os
import json
import time
from typing import List, Dict, Optional, Any
from pageindex import PageIndexClient
from groq import Groq
from dotenv import load_dotenv

class VectorlessRAGEngine:
    """
    A professional implementation of Vectorless RAG using PageIndex and Groq.
    This engine uses a tree-based retrieval strategy instead of vector embeddings.
    """

    def __init__(self, pageindex_api_key: Optional[str] = None, groq_api_key: Optional[str] = None):
        load_dotenv()
        self.pi_key = pageindex_api_key or os.getenv("PAGEINDEX_API_KEY")
        self.groq_key = groq_api_key or os.getenv("GROQ_API_KEY")

        if not self.pi_key:
            raise ValueError("PAGEINDEX_API_KEY not found.")
        if not self.groq_key:
            raise ValueError("GROQ_API_KEY not found.")

        self.pi_client = PageIndexClient(api_key=self.pi_key)
        self.groq_client = Groq(api_key=self.groq_key)

    def upload_and_index(self, pdf_path: str) -> str:
        """Uploads a PDF and waits for indexing to complete."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")

        print(f"📤 Uploading: {pdf_path}")
        result = self.pi_client.submit_document(pdf_path)
        doc_id = result["doc_id"]

        print(f"⏳ Indexing document {doc_id}...")
        while True:
            status_info = self.pi_client.get_document(doc_id)
            status = status_info.get("status")
            if status == "completed":
                print("✅ Indexing complete.")
                break
            elif status == "failed":
                raise RuntimeError("Document indexing failed.")
            time.sleep(5)
        
        return doc_id

    def get_tree(self, doc_id: str) -> List[Dict]:
        """Retrieves the hierarchical tree structure of the document."""
        result = self.pi_client.get_tree(doc_id, node_summary=True)
        return result.get("result", [])

    def _compress_tree(self, nodes: List[Dict], max_depth: int = 2, current_depth: int = 0) -> List[Dict]:
        """Compresses the tree into a minimal format for LLM reasoning."""
        compressed = []
        for n in nodes:
            entry = {
                "id": n.get("node_id"),
                "title": n.get("title"),
                "page": n.get("page_index"),
                "summary": (n.get("text") or "")[:150]
            }
            if n.get("nodes") and current_depth < max_depth:
                entry["children"] = self._compress_tree(n["nodes"], max_depth, current_depth + 1)
            compressed.append(entry)
        return compressed

    def _safe_groq_call(self, prompt: str, model: str = "llama-3.3-70b-versatile", max_retries: int = 3) -> Optional[str]:
        """Wrapper for Groq API calls with retries and timeout handling."""
        for attempt in range(max_retries):
            try:
                response = self.groq_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    timeout=30
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"⚠️ Attempt {attempt+1} failed: {e}")
                time.sleep(2)
        return None

    def tree_search(self, query: str, tree: List[Dict], expert_rules: str = "") -> Dict[str, Any]:
        """Uses LLM to identify relevant node IDs from the tree."""
        compressed = self._compress_tree(tree)
        
        prompt = f"""
You are an expert document analyzer. Identify the most relevant sections to answer the query.

Query:
{query}

Document Structure:
{json.dumps(compressed, indent=2)}

Routing Rules:
{expert_rules}

Return ONLY valid JSON:
{{
  "thinking": "Brief reasoning for your choices",
  "node_list": ["node_id1", "node_id2"]
}}
"""
        output = self._safe_groq_call(prompt)
        if not output:
            return {"thinking": "LLM call failed", "node_list": []}

        try:
            return json.loads(output)
        except json.JSONDecodeError:
            # Fallback for non-JSON responses
            import re
            match = re.search(r'\{.*\}', output, re.DOTALL)
            if match:
                try: return json.loads(match.group())
                except: pass
            return {"thinking": output, "node_list": []}

    def _find_nodes_by_ids(self, tree: List[Dict], target_ids: List[str]) -> List[Dict]:
        """Recursively finds nodes in the tree matching the target IDs."""
        found = []
        for node in tree:
            if node.get("node_id") in target_ids:
                found.append(node)
            if node.get("nodes"):
                found.extend(self._find_nodes_by_ids(node["nodes"], target_ids))
        return found

    def generate_answer(self, query: str, nodes: List[Dict]) -> str:
        """Generates a grounded answer using the content of retrieved nodes."""
        if not nodes:
            return "I couldn't find relevant sections to answer your query."

        context_blocks = []
        for n in nodes:
            context_blocks.append(
                f"### Section: {n.get('title')} (Page {n.get('page_index')})\n"
                f"{n.get('text', 'Content missing')[:2000]}"
            )
        
        context = "\n\n".join(context_blocks)
        prompt = f"""
You are an expert. Answer the question using ONLY the context below. 
Cite the section title and page number for every claim.

Question: {query}

Context:
{context}

Answer:
"""
        return self._safe_groq_call(prompt) or "Failed to generate answer."

    def run_pipeline(self, query: str, doc_id: str, expert_rules: str = "") -> Dict[str, Any]:
        """Executes the full RAG pipeline."""
        tree = self.get_tree(doc_id)
        search_result = self.tree_search(query, tree, expert_rules)
        nodes = self._find_nodes_by_ids(tree, search_result.get("node_list", []))
        answer = self.generate_answer(query, nodes)
        
        return {
            "query": query,
            "thinking": search_result.get("thinking"),
            "node_list": search_result.get("node_list"),
            "sections": [n.get("title") for n in nodes],
            "answer": answer
        }
