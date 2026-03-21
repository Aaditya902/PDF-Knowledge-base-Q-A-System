import io
import os
import tempfile
import shutil
import streamlit as st

from config import (
    GOOGLE_API_KEY, AVAILABLE_MODELS, DEFAULT_MODEL,
    SIMILARITY_THRESHOLD, TOP_K_RESULTS
)
from models.document_processor import DocumentProcessor, MAX_TOKENS
from models.knowledge_retriever import KnowledgeRetriever
from models.qa_engine import QAEngine
from utils.helpers import get_confidence_color, display_retrieved_context

st.set_page_config(
    page_title="PDF Q&A with Gemini",
    page_icon="📚",
    layout="wide"
)


def get_session_dir() -> str:
    """Return a per-session temp directory, creating it if needed."""
    if 'session_dir' not in st.session_state:
        st.session_state['session_dir'] = tempfile.mkdtemp(prefix="pdf_qa_")
    return st.session_state['session_dir']


def get_session_pdf_path() -> str:
    """Return a unique PDF path scoped to this session."""
    return os.path.join(get_session_dir(), "upload.pdf")


def cleanup_session_dir():
    """Remove the session temp directory and all its contents."""
    session_dir = st.session_state.get('session_dir')
    if session_dir and os.path.exists(session_dir):
        shutil.rmtree(session_dir, ignore_errors=True)



def initialize_session_state():
    defaults = {
        'last_results': [],
        'query': '',
        'retriever': None,
        'qa_engine': None,
        'session_dir': None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val



def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Configuration")

        selected_model = st.selectbox(
            "Select Model",
            AVAILABLE_MODELS,
            index=0
        )

        st.subheader("📊 Current Settings")
        st.write(f"Max tokens per chunk: {MAX_TOKENS}")
        st.write(f"Similarity threshold: {SIMILARITY_THRESHOLD}")
        st.write(f"Top K results: {TOP_K_RESULTS}")

        if st.button("🔄 Clear Session"):
            cleanup_session_dir()
            st.cache_resource.clear()
            st.session_state.clear()
            st.rerun()

        return selected_model


def process_document(uploaded_file, selected_model):
    temp_path = get_session_pdf_path()

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"✅ Uploaded: {uploaded_file.name}")

    try:
        with st.spinner("🔄 Processing PDF..."):
            text_content = DocumentProcessor.extract_text_from_pdf(temp_path)

            if not text_content:
                st.error("No text could be extracted from the PDF.")
                return None, None

            chunks = DocumentProcessor.chunk_text(text_content)

            retriever = KnowledgeRetriever()
            retriever.build_index(chunks)


            qa_engine = QAEngine(retriever, selected_model)

        st.success(f"✅ Processed {len(chunks)} document chunks")

        with st.expander("📖 Document Preview (First 3 chunks)"):
            for i, chunk in enumerate(chunks[:3]):
                st.markdown(f"**Chunk {i+1}:**")
                st.markdown(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                st.markdown("---")

        return retriever, qa_engine

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)
        return None, None


def render_qa_section(retriever, qa_engine):
    st.header("💬 Ask Questions")

    st.markdown("**Try asking:**")
    example_questions = [
        "What is this document about?",
        "What are the main topics discussed?",
        "Summarize the key points",
    ]

    for eq in example_questions:
        if st.button(f"📌 {eq}", key=eq):
            st.session_state['query'] = eq
            st.rerun()

    user_query = st.text_area(
        "Enter your question:",
        height=100,
        value=st.session_state.get('query', ''),
        placeholder="e.g., What is the main topic of this document?"
    )

    if st.button("🔍 Get Answer", type="primary", use_container_width=True):
        if user_query.strip():
            with st.spinner("🤔 Thinking with Gemini..."):
                response, confidence, results = qa_engine.generate_answer(user_query)

            # State update belongs here, not inside the model
            st.session_state['last_results'] = results

            st.markdown("### 📝 Answer")
            st.markdown(response)

            color = get_confidence_color(confidence)
            st.markdown(f"**Confidence:** :{color}[{confidence:.2f}]")

            display_retrieved_context(results)
        else:
            st.warning("Please enter a question.")


def main():
    st.title("📚 PDF Knowledge Base Q&A")
    st.markdown("---")

    initialize_session_state()

    if not GOOGLE_API_KEY:
        st.error("⚠️ Google API key not found. Set GOOGLE_API_KEY in your .env file or Streamlit secrets.")
        st.stop()

    selected_model = render_sidebar()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📄 Document Upload")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file is not None:
            retriever, qa_engine = process_document(uploaded_file, selected_model)

            if retriever and qa_engine:
                st.session_state['retriever'] = retriever
                st.session_state['qa_engine'] = qa_engine

    with col2:
        if st.session_state['retriever'] and st.session_state['qa_engine']:
            render_qa_section(
                st.session_state['retriever'],
                st.session_state['qa_engine']
            )
        else:
            st.info("👈 Upload a PDF to start asking questions")
            st.markdown("### 💡 Tips for better results:")
            st.markdown("""
            - Upload documents with clear text (not scanned images)
            - Ask specific questions about the content
            - Try different phrasings if you don't get good results
            - Check the retrieved context to see what was found
            """)


if __name__ == "__main__":
    main()
