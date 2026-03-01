import numpy as np
import faiss
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import streamlit as st

from config import EMBEDDING_MODEL_NAME, SIMILARITY_THRESHOLD, TOP_K_RESULTS

class KnowledgeRetriever:

    def __init__(self):
        self.index = None
        self.chunks = []
        self.embeddings = []
        self.embedding_model = self._load_embedding_model()

    @staticmethod
    @st.cache_resource
    def _load_embedding_model():
        return SentenceTransformer(EMBEDDING_MODEL_NAME)

    def build_index(self, chunks: List[str]):
        self.chunks = chunks
        self.embeddings = np.array(self._get_embeddings(chunks))
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)
        
        # Store for debugging
        st.session_state['total_chunks'] = len(chunks)

    def query(self, text: str, k: int = TOP_K_RESULTS) -> List[Tuple[str, float]]:
        if not self.index:
            return []
        
        query_embedding = np.array([self._get_embedding(text)])
        distances, indices = self.index.search(query_embedding, min(k, len(self.chunks)))
        
        results = []
        for i, index in enumerate(indices[0]):
            if index != -1 and index < len(self.chunks):
                similarity = 1 / (1 + distances[0][i])
                if similarity >= SIMILARITY_THRESHOLD:
                    results.append((self.chunks[index], similarity))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self.embedding_model.encode(texts).tolist()

    def _get_embedding(self, text: str) -> List[float]:
        return self._get_embeddings([text])[0]