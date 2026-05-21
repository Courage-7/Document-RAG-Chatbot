import streamlit as st
from sentence_transformers import SentenceTransformer

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(texts):
    model = load_model()
    if isinstance(texts, str):
        texts = [texts]
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings.tolist()