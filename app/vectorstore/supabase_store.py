import streamlit as st
import os
from supabase import create_client
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from app.config.settings import (SUPABASE_KEY, SUPABASE_URL)


load_dotenv()
supabase_client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(texts):
    model = load_model()
    if isinstance(texts, str):
        texts = [texts]
        return model.encode(texts, normalize_embeddings=True).tolist()
def retrieve_chunks(question):
    question_embedding = get_embedding(question)[0]
    response = supabase_client.rpc(
        "match_documents",
        {
            "query_embedding":
                question_embedding,
                "match_threshold":
                    0.5,
                    "match_count":
                        5
        }
    ).execute()
    return response.data  
def store_embeddings(chunks, embeddings):
    data = []
    for chunk, embedding in zip(chunks, embeddings):
        data.append({
            "content": chunk,
            "embedding": embedding
        })

    supabase_client.table("documents").insert(data).execute()  