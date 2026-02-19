import streamlit as st
from typing import List, Tuple

def get_confidence_color(confidence: float) -> str:
    """Return color based on confidence score."""
    if confidence > 0.5:
        return "green"
    elif confidence > 0.3:
        return "orange"
    else:
        return "red"

def display_retrieved_context(results: List, max_length: int = 500):
    """Display retrieved context in expander."""
    if results:
        with st.expander(f"🔍 Retrieved Context ({len(results)} chunks found)"):
            for i, (chunk, score) in enumerate(results, 1):
                st.markdown(f"**Chunk {i}** (similarity: {score:.2f})")
                display_text = chunk[:max_length] + "..." if len(chunk) > max_length else chunk
                st.markdown(display_text)
                st.markdown("---")
    else:
        with st.expander("🔍 Debug Info"):
            st.write("No chunks were retrieved. Try:")
            st.write("1. Rephrasing your question")
            st.write("2. Using simpler language")
            st.write("3. Asking about specific topics in the document")

def cleanup_temp_file(file_path: str):
    """Clean up temporary file."""
    import os
    if os.path.exists(file_path):
        os.remove(file_path)