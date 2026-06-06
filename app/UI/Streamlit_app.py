import streamlit as st
import os
import tempfile
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (PyPDFLoader,Docx2txtLoader,TextLoader)
from openai import OpenAI
from embeddings.embedding_model import get_embedding


# Load Environment Variables
load_dotenv()


# Streamlit Config
st.set_page_config(page_title="RAG Chatbot",layout="wide")


# Custom CSS
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


# Environment Variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Validate Keys
if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Missing SUPABASE_URL or SUPABASE_KEY in .env")
    st.stop()

if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY in .env")
    st.stop()
    
if "supabase_client" not in st.session_state:
    try:
        st.session_state.supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        st.error(f"Failed to create supabase client:{e}")
        st.stop()
supabase = st.session_state.supabase_client

client = OpenAI(
    api_key=OPENAI_API_KEY
)


# Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "doc_info" not in st.session_state:
    st.session_state.doc_info = None


# Sidebar
with st.sidebar:
    st.markdown("## Upload Document")

uploaded_file = st.file_uploader(
        "Upload",
        type=["pdf", "docx", "txt"],
        label_visibility="collapsed"
    )

if uploaded_file and st.session_state.doc_info is None:

        if st.button(
            "Process and Save to Supabase",
            use_container_width=True,
            type="primary"
        ):

            with st.spinner("Processing document..."):

                try:
                    # Save Temp File
                    suffix = os.path.splitext(uploaded_file.name)[1]

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=suffix
                    ) as tmp:

                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    # Load Document
                    if suffix == ".pdf":
                        loader = PyPDFLoader(tmp_path)

                    elif suffix == ".docx":
                        loader = Docx2txtLoader(tmp_path)

                    else:
                        loader = TextLoader(tmp_path)

                    docs = loader.load()

                    # Delete temp file
                    os.unlink(tmp_path)

                    # Extract Text
                    full_text = "\n".join(
                        [doc.page_content for doc in docs]
                    )

                    
                    # Create Chunks
                    chunks = text_splitter.split_text(full_text)

                    if not chunks:
                        st.error("No text chunks were created.")
                        st.stop()

                    
                    # Generate Embedding
                    embeddings = get_embedding(chunks)

                    # Prepare Data
                    data = []

                    for chunk, embedding in zip(chunks, embeddings):

                        # Convert numpy array to list if needed
                        if hasattr(embedding, "tolist"):
                            embedding = embedding.tolist()

                        data.append({
                            "content": chunk,
                            "embedding": embedding,
                            "metadata": {
                                "filename": uploaded_file.name
                            }
                        })

                    
                    # Insert into Supabase
                    supabase.table("documnets").insert(data).execute()
                    
                    # Save Document Info
                    st.session_state.doc_info = {
                        "filename": uploaded_file.name,
                        "pages": len(docs),
                        "chunks": len(chunks),
                        "uploaded_at": datetime.now().strftime(
                            "%b %d, %Y %I:%M %p"
                        )
                    }

                    st.success("Document processed successfully!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # Document Info
        if st.session_state.doc_info:
            st.markdown("## Document Info")
            

        st.write(
            f"Filename: {st.session_state.doc_info['filename']}"
        )

        st.write(
            f"Pages: {st.session_state.doc_info['pages']}"
        )

        st.write(
            f"Chunks: {st.session_state.doc_info['chunks']}"
        )

        st.write(
            f"Uploaded: {st.session_state.doc_info['uploaded_at']}"
        )

   
# Settings
st.markdown("## Settings")

top_k = st.slider(
        "Top K Chunks",
        1,
        10,
        5
    )

model = st.selectbox(
        "Model",
        [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-3.5-turbo"
        ]
    )

show_sources = st.toggle(
    "Show Sources",
    value=True
)

# Main Header

st.markdown("# EAZI'S RAG Chatbot")

st.caption(
    "Upload a document and ask questions about it."
)


# Clear Chat
top_col1, top_col2 = st.columns([6, 1])

with top_col2:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# Display Chat History
for msg in st.session_state.messages:

    if msg["role"] == "user":

        with st.chat_message("user", avatar="👤"):

            st.markdown(
                f"<div class='user-msg'>{msg['content']}</div>",
                unsafe_allow_html=True
            )

            st.caption(msg["time"])

    else:

        with st.chat_message("assistant", avatar="🤖"):

            st.markdown(
                f"<div class='bot-msg'>{msg['content']}</div>",
                unsafe_allow_html=True
            )

            if show_sources and "sources" in msg:

                with st.expander("Sources"):

                    for s in msg["sources"]:
                        st.markdown(f"- {s}")

            st.caption(msg["time"])


# Chat Input
if prompt := st.chat_input(
    "Ask a question about your document..."
):

    if not st.session_state.doc_info:

        st.warning(
            "Please upload and process a document first."
        )

    else:

        # Add User Message
        time_now = datetime.now().strftime("%I:%M %p")

        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "time": time_now
        })

        with st.spinner("Searching document..."):

            try:
                # Query Embedding
                query_embedding = get_embedding([prompt])[0]

                if hasattr(query_embedding, "tolist"):
                    query_embedding = query_embedding.tolist()

                # Search Supabase
                result = supabase.rpc(
                    "match_documents",
                    {
                        "query_embedding": query_embedding,
                        "match_count": top_k
                    }
                ).execute()

                if not result.data:
                    answer = "No relevant information found."
                    sources = []

                else:
                    # Build Context
                    context = "\n\n".join([
                        r["content"]
                        for r in result.data
                    ])

                    sources = [
                        r["metadata"]["filename"]
                        for r in result.data
                    ]
                    # OpenAI Response
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "Answer ONLY using the provided context. "
                                    "If the answer is not in the context, say you don't know."
                                )
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"Context:\n{context}\n\n"
                                    f"Question: {prompt}"
                                )
                            }
                        ],
                        temperature=0.2
                    )

                    answer = response.choices[0].message.content

            except Exception as e:
                answer = f"Error: {str(e)}"
                sources = []
        # Save Assistant Message
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "time": datetime.now().strftime("%I:%M %p")
        })

        st.rerun()

# Footer
st.divider()

st.caption(
    "RAG Chatbot built with Streamlit • Supabase • OpenAI"
)