import streamlit as st
import os

from loaders import load_pdf
from loaders import load_docx
from loaders import load_image

from utils import split_text
from embeddings import get_embeddings
from vectorstore import supabase_store
from rag import chatbot

st.title("Document RAG Chatbot")

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf", "docx", "png", "jpg", "jpeg"]
)

if uploaded_file:

    save_path = f"data/uploads/{uploaded_file.name}"

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    extension = uploaded_file.name.split(".")[-1]

    if extension == "pdf":
        text = load_pdf(save_path)

    elif extension == "docx":
        text = load_docx(save_path)

    else:
        text = load_image(save_path)

    chunks = split_text(text)

    embeddings = get_embeddings(chunks)

    supabase_store.store_embeddings(chunks, embeddings)

    st.success("Document processed successfully!")

question = st.text_input("Ask a question about the document")

if question:
    answer = chatbot.ask_chatbot(question)

    st.write(answer)