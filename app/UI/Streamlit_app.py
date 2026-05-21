import streamlit as st
import os
import tempfile
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from openai import OpenAI

from embeddings.embedding_model import get_embedding

load_dotenv()

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# Custom CSS for chat bubbles
st.markdown("""
<style>
.user-msg {
    background-color: #fee2e2;
    padding: 12px 16px;
    border-radius: 12px;
    margin-bottom: 8px;
}
.bot-msg {
    background-color: #eff6ff;
    padding: 12px 16px;
    border-radius: 12px;
    margin-bottom: 8px;
}
.stChatMessage {
    padding: 0;
}
</style>
""", unsafe_allow_html=True)

# --- Init ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Missing SUPABASE_URL or SUPABASE_KEY in.env")
    st.stop()

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_info" not in st.session_state:
    st.session_state.doc_info = None

#   Sidebar 
with st.sidebar:
    st.markdown("📄 Upload Document")
    st.caption("Upload a PDF, Word document, or TXT file.")

    uploaded_file = st.file_uploader(
        "Upload",
        type=["pdf", "docx", "txt"],
        label_visibility="collapsed"
    )

    if uploaded_file and st.session_state.doc_info is None:
        if st.button("Process and Save to Supabase", use_container_width=True, type="primary"):
            with st.spinner("Processing..."):
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                if suffix == ".pdf":
                    loader = PyPDFLoader(tmp_path)
                elif suffix == ".docx":
                    loader = Docx2txtLoader(tmp_path)
                else:
                    loader = TextLoader(tmp_path)

                docs = loader.load()
                os.unlink(tmp_path)

                chunks = text_splitter.split_text("\n".join([d.page_content for d in docs]))
                embeddings = get_embedding(chunks)

                data = [
                    {"content": c, "embedding": e, "metadata": {"filename": uploaded_file.name}}
                    for c, e in zip(chunks, embeddings)
                ]
                supabase.table("documents").insert(data).execute()

                st.session_state.doc_info = {
                    "filename": uploaded_file.name,
                    "pages": len(docs),
                    "chunks": len(chunks),
                    "uploaded_at": datetime.now().strftime("%b %d, %Y %I:%M %p")
                }
            st.success("✅ Document processed!")

    if st.session_state.doc_info:
        st.markdown("📋 Document Info")
        st.write(f"Filename: {st.session_state.doc_info['filename']}")
        st.write(f"Pages: {st.session_state.doc_info['pages']}")
        st.write(f"Chunks: {st.session_state.doc_info['chunks']}")
        st.write(f"Uploaded: {st.session_state.doc_info['uploaded_at']}")

    st.markdown("⚙️ Settings")
    top_k = st.slider("Top K Chunks", 1, 10, 5)
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
    show_sources = st.toggle("Show Sources", value=True)

# Main Chat Area
col1, col2 = st.columns([1, 20])
with col1:
    st.markdown("🤖")
with col2:
    st.markdown("EAZI'S RAG Chatbot")
    st.caption("Upload a document and ask questions about it. I'll answer based on the content.")

# Clear chat button top right
top_col1, top_col2 = st.columns([6, 1])
with top_col2:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(f"<div class='user-msg'>{msg['content']}</div>", unsafe_allow_html=True)
            st.caption(msg["time"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(f"<div class='bot-msg'>{msg['content']}</div>", unsafe_allow_html=True)

            if show_sources and "sources" in msg:
                with st.expander("📚 Sources"):
                    for s in msg["sources"]:
                        st.markdown(f"- {s}")

            st.caption(msg["time"])

# Chat input
if prompt := st.chat_input("Ask a question about your document..."):
    if not st.session_state.doc_info:
        st.warning("Please upload and process a document first.")
    elif not OPENAI_API_KEY:
        st.error("Missing OPENAI_API_KEY in.env")
    else:
        # Add user message
        time_now = datetime.now().strftime("%I:%M %p")
        st.session_state.messages.append({"role": "user", "content": prompt, "time": time_now})

        # Get embedding and search Supabase
        with st.spinner("Searching..."):
            query_embedding = get_embedding(prompt)[0]
            result = supabase.rpc("match_documents", {
                "query_embedding": query_embedding,
                "match_count": top_k
            }).execute()

            context = "\n\n".join([r["content"] for r in result.data])
            sources = [f"{r['metadata']['filename']}" for r in result.data]

            # Call OpenAI
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Answer based only on the context provided. If the answer isn't in the context, say you don't know."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}"}
                ],
                temperature=0.2
            )
            answer = response.choices[0].message.content

        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "time": datetime.now().strftime("%I:%M %p")
        })
        st.rerun()

# Footer
st.divider()
st.caption("RAG Chatbot built with Streamlit • Python • Supabase • OpenAI")