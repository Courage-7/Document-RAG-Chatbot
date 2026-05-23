import streamlit as st
from sentence_transformers import SentenceTransformer

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(texts):
    model = load_model()
    return model.encode(texts).tolist()