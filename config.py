import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    try:
        GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        pass  

AVAILABLE_MODELS = [
    "models/gemini-2.5-flash",
    #"models/gemini-2.5-pro", 
    #"models/gemini-2.0-flash",
]

DEFAULT_MODEL = "models/gemini-2.5-flash"

SIMILARITY_THRESHOLD = 0.3
TOP_K_RESULTS = 5

EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

TEMPERATURE = 0.3
TOP_P = 0.9
MAX_OUTPUT_TOKENS = 1024