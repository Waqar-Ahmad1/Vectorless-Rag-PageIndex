import streamlit as st
import os
from rag_engine import VectorlessRAGEngine
from dotenv import load_dotenv

# Page config
st.set_page_config(page_title="Vectorless RAG Chatbot", page_icon="🧬", layout="wide")

# App Styles
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stChatFloatingInputContainer {
        padding-bottom: 20px;
    }
    .st-emotion-cache-1c7n2ka {
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Initialization
if "engine" not in st.session_state:
    try:
        st.session_state.engine = VectorlessRAGEngine()
    except Exception as e:
        st.error(f"Initialization Error: {e}")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "doc_id" not in st.session_state:
    st.session_state.doc_id = None

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")
    uploaded_file = st.file_uploader("Upload PDF Document", type="pdf")
    
    if uploaded_file:
        # Save file to disk
        temp_path = os.path.join("/tmp", uploaded_file.name) if os.name != 'nt' else os.path.join(os.environ.get('TEMP', '.'), uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("Index Document"):
            with st.spinner("Uploading and indexing... this may take a minute."):
                try:
                    doc_id = st.session_state.engine.upload_and_index(temp_path)
                    st.session_state.doc_id = doc_id
                    st.success(f"Document Indexed! ID: {doc_id}")
                except Exception as e:
                    st.error(f"Indexing Failed: {e}")

    st.markdown("---")
    expert_rules = st.text_area("Expert Routing Rules", 
        placeholder="e.g. If query mentions EBITDA -> prioritize MD&A section",
        height=200)

# Main Interface
st.title("🧬 Vectorless RAG Chatbot")
st.caption("Reasoning-based RAG with Tree Indexing (No Vector DB)")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "thinking" in message:
            with st.expander("Show Reasoning"):
                st.info(message["thinking"])

# Chat Input
if prompt := st.chat_input("Ask a question about your document..."):
    if not st.session_state.doc_id:
        st.warning("Please upload and index a document first.")
    else:
        # User message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Bot response
        with st.chat_message("assistant"):
            with st.spinner("Reasoning..."):
                try:
                    result = st.session_state.engine.run_pipeline(
                        query=prompt, 
                        doc_id=st.session_state.doc_id, 
                        expert_rules=expert_rules
                    )
                    
                    answer = result["answer"]
                    thinking = result["thinking"]
                    
                    st.markdown(answer)
                    with st.expander("Show Reasoning"):
                        st.info(thinking)
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer, 
                        "thinking": thinking
                    })
                except Exception as e:
                    st.error(f"Pipeline Error: {e}")
