import streamlit as st
import numpy as np
import faiss
import PyPDF2
import os
from google import genai
from dotenv import load_dotenv
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

load_dotenv()

# Configure Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    client = genai.Client(api_key=GOOGLE_API_KEY)

# ADJUSTED: Lower thresholds for better retrieval
MAX_CHUNK_SIZE = 1000  # Increased chunk size
SIMILARITY_THRESHOLD = 0.3  # Lowered threshold
TOP_K_RESULTS = 5  # Get more results

# Load local embedding model
@st.cache_resource
def load_embedding_model():
    """Cache the embedding model to avoid reloading"""
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

class DocumentProcessor:
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """Extract text content from PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = []
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        # Add page number for reference
                        text.append(f"[Page {page_num + 1}]\n{page_text}")
                return '\n\n'.join(text).strip()
        except Exception as e:
            raise RuntimeError(f"PDF extraction failed: {str(e)}")

    @staticmethod
    def chunk_text(text: str, max_size: int = MAX_CHUNK_SIZE) -> List[str]:
        """Split text into chunks with overlap for better context."""
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_length = len(para)
            
            # If single paragraph is too long, split by sentences
            if para_length > max_size:
                sentences = para.split('. ')
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    if not sentence.endswith('.'):
                        sentence += '.'
                    
                    if len(sentence) > max_size:
                        # If sentence is still too long, truncate
                        chunks.append(sentence[:max_size])
                    else:
                        if current_length + len(sentence) > max_size and current_chunk:
                            chunks.append(' '.join(current_chunk))
                            current_chunk = []
                            current_length = 0
                        current_chunk.append(sentence)
                        current_length += len(sentence) + 1
            else:
                # Paragraph fits, add to current chunk
                if current_length + para_length > max_size and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                current_chunk.append(para)
                current_length += para_length + 2  # Account for \n\n
        
        # Add last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # Create overlapping chunks for better retrieval
        overlapping_chunks = []
        for i in range(len(chunks)):
            if i > 0:
                # Combine with previous chunk for overlap
                combined = chunks[i-1] + " " + chunks[i]
                if len(combined) <= max_size * 1.5:
                    overlapping_chunks.append(combined)
            overlapping_chunks.append(chunks[i])
        
        return overlapping_chunks

class KnowledgeRetriever:
    def __init__(self):
        self.index = None
        self.chunks = []
        self.embeddings = []

    def build_index(self, chunks: List[str]):
        """Create FAISS index from document chunks."""
        self.chunks = chunks
        self.embeddings = np.array(self._get_embeddings(chunks))
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)
        
        # Store for debugging
        st.session_state['total_chunks'] = len(chunks)

    def query(self, text: str, k: int = TOP_K_RESULTS) -> List[Tuple[str, float]]:
        """Retrieve relevant document chunks."""
        query_embedding = np.array([self._get_embedding(text)])
        distances, indices = self.index.search(query_embedding, min(k, len(self.chunks)))
        
        results = []
        for i, index in enumerate(indices[0]):
            if index != -1 and index < len(self.chunks):
                # Convert L2 distance to similarity score (0-1)
                similarity = 1 / (1 + distances[0][i])
                # Lower threshold to get more results
                if similarity >= SIMILARITY_THRESHOLD:
                    results.append((self.chunks[index], similarity))
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts."""
        return embedding_model.encode(texts).tolist()

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        return self._get_embeddings([text])[0]

class QAEngine:
    def __init__(self, retriever: KnowledgeRetriever):
        self.retriever = retriever
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.model_name = "models/gemini-2.5-flash"
    
    def generate_answer(self, query: str) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Generate answer using Gemini."""
        try:
            # Retrieve relevant context
            results = self.retriever.query(query)
            
            # Debug: Store results in session state
            st.session_state['last_results'] = results
            
            if not results:
                # If no results with threshold, still return top result with low confidence
                results = self.retriever.query(query, k=1)
                if results:
                    return self._generate_with_context(results, query, results[0][1])
                else:
                    return "I couldn't find relevant information in the document. Please try rephrasing your question or the document might not contain this information.", 0.0, []

            return self._generate_with_context(results, query, sum(r[1] for r in results) / len(results))
            
        except Exception as e:
            return f"Error generating response: {str(e)}", 0.0, []

    def _generate_with_context(self, results: List[Tuple[str, float]], query: str, avg_similarity: float) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Generate answer with given context."""
        # Combine contexts with weights
        context_parts = []
        for i, (chunk, score) in enumerate(results[:3]):  # Use top 3 chunks
            context_parts.append(f"[Context {i+1} (Relevance: {score:.2f})]:\n{chunk}")
        
        context = '\n\n'.join(context_parts)
        
        # Create prompt
        prompt = f"""You are a helpful assistant that answers questions based on the provided document context.

CONTEXT FROM DOCUMENT:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
1. Answer the question based ONLY on the provided context
2. If the context doesn't contain the exact answer, use the most relevant information available
3. If the information is completely absent, say "Based on the document, I cannot find information about [topic]"
4. Be specific and quote relevant parts when possible
5. Keep your answer concise but informative

ANSWER:"""

        # Generate response
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={
                "temperature": 0.3,  # Lower temperature for more focused answers
                "top_p": 0.9,
                "max_output_tokens": 1024,
            }
        )
        
        return response.text, avg_similarity, results

# Streamlit UI
def main():
    st.set_page_config(
        page_title="PDF Q&A with Gemini",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 PDF Knowledge Base Q&A with Gemini 2.5 Flash")
    st.markdown("---")
    
    # Initialize session state
    if 'last_results' not in st.session_state:
        st.session_state['last_results'] = []
    if 'query' not in st.session_state:
        st.session_state['query'] = ''
    
    # Check if API key is configured
    if not GOOGLE_API_KEY:
        st.error("⚠️ Google API key not found. Please set GOOGLE_API_KEY in your .env file")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_options = [
            "models/gemini-2.5-flash",
            "models/gemini-2.5-pro", 
            "models/gemini-2.0-flash",
        ]
        
        selected_model = st.selectbox(
            "Select Model",
            model_options,
            index=0
        )
        
        st.subheader("📊 Current Settings")
        st.write(f"Chunk size: {MAX_CHUNK_SIZE}")
        st.write(f"Similarity threshold: {SIMILARITY_THRESHOLD}")
        st.write(f"Top K results: {TOP_K_RESULTS}")
        
        if st.button("🔄 Clear Session"):
            st.cache_resource.clear()
            st.session_state.clear()
            st.rerun()
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 Document Upload")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf"
        )
        
        if uploaded_file is not None:
            # Save temporary file
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"✅ Uploaded: {uploaded_file.name}")

            try:
                with st.spinner("🔄 Processing PDF..."):
                    # Extract text
                    text_content = DocumentProcessor.extract_text_from_pdf("temp.pdf")
                    
                    if not text_content:
                        st.error("No text could be extracted from the PDF.")
                        return
                    
                    # Create chunks
                    chunks = DocumentProcessor.chunk_text(text_content)
                    
                    # Build retrieval index
                    retriever = KnowledgeRetriever()
                    retriever.build_index(chunks)
                    
                    # Initialize QA engine
                    qa_engine = QAEngine(retriever)
                    qa_engine.model_name = selected_model

                st.success(f"✅ Processed {len(chunks)} document chunks")
                
                # Show document preview
                with st.expander("📖 Document Preview (First 3 chunks)"):
                    for i, chunk in enumerate(chunks[:3]):
                        st.markdown(f"**Chunk {i+1}:**")
                        st.markdown(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                        st.markdown("---")
                
                with col2:
                    st.header("💬 Ask Questions")
                    
                    # Example questions
                    st.markdown("**Try asking:**")
                    example_questions = [
                        "What is this document about?",
                        "What are the main topics discussed?",
                        "Summarize the key points"
                    ]
                    
                    for eq in example_questions:
                        if st.button(f"📌 {eq}", key=eq):
                            st.session_state['query'] = eq
                            st.rerun()
                    
                    # User query input
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
                            
                            # Display answer
                            st.markdown("### 📝 Answer")
                            st.markdown(response)
                            
                            # Show confidence
                            confidence_color = "green" if confidence > 0.5 else "orange" if confidence > 0.3 else "red"
                            st.markdown(f"**Confidence:** :{confidence_color}[{confidence:.2f}]")
                            
                            # Show retrieved context
                            if results:
                                with st.expander(f"🔍 Retrieved Context ({len(results)} chunks found)"):
                                    for i, (chunk, score) in enumerate(results, 1):
                                        st.markdown(f"**Chunk {i}** (similarity: {score:.2f})")
                                        st.markdown(chunk[:500] + "..." if len(chunk) > 500 else chunk)
                                        st.markdown("---")
                            else:
                                with st.expander("🔍 Debug Info"):
                                    st.write("No chunks were retrieved. Try:")
                                    st.write("1. Rephrasing your question")
                                    st.write("2. Using simpler language")
                                    st.write("3. Asking about specific topics in the document")
                        else:
                            st.warning("Please enter a question.")
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)
            finally:
                if os.path.exists("temp.pdf"):
                    os.remove("temp.pdf")
        else:
            with col2:
                st.info("👈 Upload a PDF to start asking questions")
                
                # Tips
                st.markdown("### 💡 Tips for better results:")
                st.markdown("""
                - Upload documents with clear text (not scanned images)
                - Ask specific questions about the content
                - Try different phrasings if you don't get good results
                - Check the retrieved context to see what was found
                """)

if __name__ == "__main__":
    main()