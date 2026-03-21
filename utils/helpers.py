import streamlit as st
from typing import List
import logging

logger = logging.getLogger(__name__)


def truncate(text: str, max_len: int) -> str:
    return text[:max_len] + "..." if len(text) > max_len else text


def get_confidence_color(confidence: float) -> str:
    if confidence > 0.5:
        return "green"
    elif confidence > 0.3:
        return "orange"
    else:
        return "red"


def display_retrieved_context(results: List, max_length: int = 500):
    if results:
        with st.expander(f"🔍 Retrieved Context ({len(results)} chunks found)"):
            for i, (chunk, score) in enumerate(results, 1):
                st.markdown(f"**Chunk {i}** (similarity: {score:.2f})")
                st.markdown(truncate(chunk, max_length))
                st.markdown("---")
    else:
        with st.expander("🔍 Debug Info"):
            st.write("No chunks were retrieved. Try:")
            st.write("1. Rephrasing your question")
            st.write("2. Using simpler language")
            st.write("3. Asking about specific topics in the document")