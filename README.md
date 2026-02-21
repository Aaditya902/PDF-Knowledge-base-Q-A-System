PDF Knowledge Base Q&A System (Gemini + FAISS + Streamlit)

An AI-powered document question-answering system that allows users to upload PDFs and ask questions about their content. The system uses semantic search (FAISS) and Google Gemini models to provide accurate, context-aware answers.


Key Features

Upload and process PDF documents
Smart text chunking with overlap
Semantic search using FAISS
Local embeddings with Sentence Transformers
Answer generation using Google Gemini
Confidence scoring for responses
Interactive UI built with Streamlit
Debug view for retrieved context



Project Structure


```bash
rag-gemini/
├── app.py                    # Main Streamlit app
├── config.py                 # Configuration settings
├── requirements.txt          # Dependencies
├── README.md
├── models/
│   ├── __init__.py
│   ├── document_processor.py  # PDF extraction & chunking
│   ├── knowledge_retriever.py # Embedding + FAISS
│   └── qa_engine.py           # Gemini-based QA
└── utils/
    ├── __init__.py
    └── helpers.py             # UI helpers

```

System Architecture

flowchart TD
    A[PDF Upload] --> B[Text Extraction]
    B --> C[Chunking + Overlap]
    C --> D[Embeddings]
    D --> E[FAISS Index]

    Q[User Query] --> QE[Query Embedding]
    QE --> F[Similarity Search]
    F --> G[Top-K Context]
    G --> H[Gemini LLM]
    H --> I[Answer + Confidence]



Tech Stack

| Layer        | Technology                                          | Reasoning                                 |
| ------------ | --------------------------------------------------- | ----------------------------------------- |
| UI           | Streamlit                                           | Rapid prototyping for interactive AI apps |
| LLM          | Google Gemini (`2.5-flash`, `2.5-pro`, `2.0-flash`) | Fast, cost-efficient generation           |
| RAG Pipeline | Custom implementation                               | Full control over retrieval logic         |
| Embeddings   | Sentence Transformers (`all-MiniLM-L6-v2`)          | Lightweight, low latency                  |
| Vector Store | FAISS (`IndexFlatL2`)                               | Exact search, no external dependency      |
| PDF Parsing  | PyPDF2                                              | Simple text extraction                    |
| Numerical    | NumPy                                               | Efficient vector operations               |
| API Client   | google-genai                                        | Gemini integration                        |
| State Mgmt   | Streamlit session state                             | Maintain UI state                         |
| Language     | Python                                              | Ecosystem support                         |



Setup Instructions

1. Clone Repository
git clone https://github.com/your-username/rag-gemini.git
cd rag-gemini

2. Create Virtual Environment
python -m venv myenv
source myenv/bin/activate      # Mac/Linux
myenv\Scripts\activate         # Windows

3. Install Dependencies
pip install -r requirements.txt

4. Configure Environment Variables

Create a .env file in root:

GOOGLE_API_KEY=your_google_api_key



Run the Application
streamlit run app.py

Open in browser:

http://localhost:8501



How It Works
1. Document Processing

Extracts text using PyPDF2
Adds page references

2. Chunking

Splits text into chunks (MAX_CHUNK_SIZE = 1000)
Adds overlapping context for better retrieval

3. Embedding

Uses all-MiniLM-L6-v2
Converts text into dense vectors

4. Retrieval

Stores embeddings in FAISS index
Performs similarity search

5. Answer Generation

Retrieves top-k relevant chunks
Sends context + query to Gemini
Generates response



Confidence Scoring

The system calculates confidence based on similarity scores:

| Score   | Meaning         |
| ------- | --------------- |
| > 0.5   | High confidence |
| 0.3–0.5 | Medium          |
| < 0.3   | Low             |



If you found this useful, give it a star!