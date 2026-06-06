import os
import sys
import tempfile
from datetime import datetime
from html import escape
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from supabase import create_client

from app.config.settings import (
    DEFAULT_GROQ_MODEL,
    ENV_FILE,
    GROQ_BASE_URL,
    GROQ_MODELS,
    REQUIRED_ENV_VARS,
    apply_runtime_settings,
    get_runtime_settings,
    missing_required_settings,
    save_settings_to_env,
)
from embeddings.embedding_model import get_embedding


SETTINGS_PREFIX = "credential_"


st.set_page_config(
    page_title="RAG Chatbot",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container {
    max-width: 1180px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.app-kicker {
    color: #64748b;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}

.app-title {
    color: #111827;
    font-size: 2rem;
    font-weight: 750;
    line-height: 1.15;
    margin: 0;
}

.app-subtitle {
    color: #475569;
    margin-top: 0.45rem;
    margin-bottom: 1rem;
}

.status-card {
    background: #ffffff;
    border: 1px solid #d9e2ec;
    border-left: 4px solid #64748b;
    border-radius: 8px;
    min-height: 88px;
    padding: 0.8rem 0.9rem;
}

.status-card.ready {
    border-left-color: #16a34a;
}

.status-card.warn {
    border-left-color: #f59e0b;
}

.status-card.error {
    border-left-color: #dc2626;
}

.status-card.info {
    border-left-color: #2563eb;
}

.status-label {
    color: #64748b;
    display: block;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0;
    text-transform: uppercase;
}

.status-value {
    color: #111827;
    display: block;
    font-size: 1rem;
    font-weight: 750;
    margin-top: 0.25rem;
}

.status-detail {
    color: #64748b;
    display: block;
    font-size: 0.84rem;
    margin-top: 0.2rem;
}

.sidebar-badge {
    border: 1px solid #d9e2ec;
    border-radius: 8px;
    margin-bottom: 0.75rem;
    padding: 0.65rem 0.75rem;
}

.sidebar-badge.ready {
    background: #f0fdf4;
    border-color: #bbf7d0;
    color: #166534;
}

.sidebar-badge.warn {
    background: #fffbeb;
    border-color: #fde68a;
    color: #92400e;
}

.sidebar-badge.error {
    background: #fef2f2;
    border-color: #fecaca;
    color: #991b1b;
}

.sidebar-badge-title {
    display: block;
    font-size: 0.82rem;
    font-weight: 750;
}

.sidebar-badge-detail {
    display: block;
    font-size: 0.78rem;
    margin-top: 0.1rem;
}

.chat-empty {
    background: #f8fafc;
    border: 1px dashed #cbd5e1;
    border-radius: 8px;
    color: #475569;
    margin-top: 0.5rem;
    padding: 1.1rem 1.2rem;
}

.chat-empty strong {
    color: #111827;
    display: block;
    margin-bottom: 0.2rem;
}

.chat-msg {
    border-radius: 8px;
    margin-bottom: 0.45rem;
    padding: 0.75rem 0.9rem;
}

.chat-msg.user {
    background: #f1f5f9;
    border: 1px solid #d9e2ec;
}

.chat-msg.assistant {
    background: #eef6ff;
    border: 1px solid #bfdbfe;
}

.chat-time {
    color: #94a3b8;
    display: block;
    font-size: 0.76rem;
    margin-top: 0.2rem;
}

.stChatMessage {
    padding: 0;
}
</style>
""",
    unsafe_allow_html=True,
)


def initialize_credential_state() -> None:
    env_settings = get_runtime_settings()

    for name in REQUIRED_ENV_VARS:
        st.session_state.setdefault(
            f"{SETTINGS_PREFIX}{name}",
            env_settings.get(name, ""),
        )


def get_credentials_from_state() -> dict[str, str]:
    return {
        name: st.session_state.get(f"{SETTINGS_PREFIX}{name}", "").strip()
        for name in REQUIRED_ENV_VARS
    }


def render_status_card(
    label: str,
    value: str,
    detail: str,
    tone: str,
) -> None:
    st.markdown(
        f"""
<div class="status-card {tone}">
    <span class="status-label">{escape(label)}</span>
    <span class="status-value">{escape(value)}</span>
    <span class="status-detail">{escape(detail)}</span>
</div>
""",
        unsafe_allow_html=True,
    )


def render_sidebar_badge(title: str, detail: str, tone: str) -> None:
    st.markdown(
        f"""
<div class="sidebar-badge {tone}">
    <span class="sidebar-badge-title">{escape(title)}</span>
    <span class="sidebar-badge-detail">{escape(detail)}</span>
</div>
""",
        unsafe_allow_html=True,
    )


def render_settings(settings_ready: bool) -> dict[str, str]:
    with st.expander("Settings", expanded=not settings_ready):
        with st.form("credential_settings"):
            st.text_input(
                "GROQ_API_KEY",
                type="password",
                key=f"{SETTINGS_PREFIX}GROQ_API_KEY",
            )
            st.text_input(
                "OPENROUTER_API_KEY",
                type="password",
                key=f"{SETTINGS_PREFIX}OPENROUTER_API_KEY",
            )
            st.text_input(
                "SUPABASE_URL",
                key=f"{SETTINGS_PREFIX}SUPABASE_URL",
            )
            st.text_input(
                "SUPABASE_KEY",
                type="password",
                key=f"{SETTINGS_PREFIX}SUPABASE_KEY",
            )

            save_to_env = st.checkbox("Save to .env")
            submitted = st.form_submit_button(
                "Apply Settings",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            credentials = get_credentials_from_state()
            apply_runtime_settings(credentials)

            if save_to_env:
                save_settings_to_env(credentials)
                st.success(f"Saved settings to {ENV_FILE.name}.")
            else:
                st.success("Settings applied for this session.")

            st.rerun()

    return get_credentials_from_state()


def get_supabase_client(credentials: dict[str, str]):
    client_credentials = (
        credentials["SUPABASE_URL"],
        credentials["SUPABASE_KEY"],
    )

    if (
        "supabase_client" not in st.session_state
        or st.session_state.get("supabase_client_credentials")
        != client_credentials
    ):
        st.session_state.supabase_client = create_client(
            credentials["SUPABASE_URL"],
            credentials["SUPABASE_KEY"],
        )
        st.session_state.supabase_client_credentials = client_credentials

    return st.session_state.supabase_client


def get_groq_client(credentials: dict[str, str]) -> OpenAI:
    return OpenAI(
        api_key=credentials["GROQ_API_KEY"],
        base_url=GROQ_BASE_URL,
    )


def load_uploaded_documents(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif suffix == ".docx":
            loader = Docx2txtLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path)

        return loader.load()
    finally:
        os.unlink(tmp_path)


def render_message(message: dict[str, str], show_sources: bool) -> None:
    role = message["role"]
    css_role = "user" if role == "user" else "assistant"

    with st.chat_message(role):
        safe_content = escape(message["content"]).replace("\n", "<br>")
        st.markdown(
            f"<div class='chat-msg {css_role}'>{safe_content}</div>",
            unsafe_allow_html=True,
        )

        if show_sources and role == "assistant" and "sources" in message:
            sources = message["sources"]
            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.markdown(f"- {escape(str(source))}")

        st.markdown(
            f"<span class='chat-time'>{escape(message['time'])}</span>",
            unsafe_allow_html=True,
        )


initialize_credential_state()
credentials = get_credentials_from_state()
apply_runtime_settings(credentials)
missing_settings = missing_required_settings(credentials)
settings_ready = not missing_settings

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "doc_info" not in st.session_state:
    st.session_state.doc_info = None

supabase = None
connection_error = None

if settings_ready:
    try:
        supabase = get_supabase_client(credentials)
    except Exception as exc:
        settings_ready = False
        connection_error = str(exc)

with st.sidebar:
    st.markdown("## RAG Chatbot")

    if os.getenv("OPENAI_API_KEY"):
        st.caption(
            "Legacy OPENAI_API_KEY found in .env. "
            "This app now uses OPENROUTER_API_KEY for embeddings."
        )

    if settings_ready:
        render_sidebar_badge(
            "Settings ready",
            "Groq, OpenRouter, and Supabase credentials are available.",
            "ready",
        )
    elif connection_error:
        render_sidebar_badge(
            "Connection issue",
            connection_error,
            "error",
        )
    else:
        render_sidebar_badge(
            "Settings incomplete",
            "Missing: " + ", ".join(missing_settings),
            "warn",
        )

    st.divider()

    st.markdown("### Document")

    uploaded_file = st.file_uploader(
        "Upload document",
        type=["pdf", "docx", "txt"],
        disabled=not settings_ready,
    )

    process_disabled = (
        not settings_ready
        or uploaded_file is None
        or st.session_state.doc_info is not None
    )
    process_clicked = st.button(
        "Process and Save",
        use_container_width=True,
        type="primary",
        disabled=process_disabled,
    )

    if uploaded_file and st.session_state.doc_info is not None:
        st.caption("Clear the current document before processing another one.")

    if (
        uploaded_file
        and st.session_state.doc_info is None
        and process_clicked
        and supabase is not None
    ):
        with st.spinner("Processing document..."):
            try:
                docs = load_uploaded_documents(uploaded_file)
                full_text = "\n".join(doc.page_content for doc in docs)
                chunks = text_splitter.split_text(full_text)

                if not chunks:
                    st.error("No text chunks were created.")
                else:
                    embeddings = get_embedding(chunks)

                    data = []

                    for chunk, embedding in zip(chunks, embeddings):
                        if hasattr(embedding, "tolist"):
                            embedding = embedding.tolist()

                        data.append(
                            {
                                "content": chunk,
                                "embedding": embedding,
                                "metadata": {
                                    "filename": uploaded_file.name,
                                },
                            }
                        )

                    supabase.table("documents").insert(data).execute()

                    st.session_state.doc_info = {
                        "filename": uploaded_file.name,
                        "pages": len(docs),
                        "chunks": len(chunks),
                        "uploaded_at": datetime.now().strftime(
                            "%b %d, %Y %I:%M %p"
                        ),
                    }

                    st.success("Document processed successfully.")

            except Exception as exc:
                st.error(f"Error: {exc}")

    if st.session_state.doc_info:
        doc_info = st.session_state.doc_info
        st.markdown(
            f"**File:** {escape(str(doc_info['filename']))}",
            unsafe_allow_html=True,
        )
        st.caption(
            f"{doc_info['pages']} pages, {doc_info['chunks']} chunks"
        )
        st.caption(f"Uploaded {doc_info['uploaded_at']}")

        if st.button("Clear Document", use_container_width=True):
            st.session_state.doc_info = None
            st.rerun()
    else:
        st.caption("No document loaded.")

    st.divider()

    st.markdown("### Retrieval")

    top_k = st.slider(
        "Top K Chunks",
        1,
        10,
        5,
        disabled=not settings_ready,
    )

    model = st.selectbox(
        "Groq Model",
        GROQ_MODELS,
        index=GROQ_MODELS.index(DEFAULT_GROQ_MODEL),
        disabled=not settings_ready,
    )

    show_sources = st.toggle(
        "Show Sources",
        value=True,
    )

    st.divider()

    credentials = render_settings(settings_ready)
    apply_runtime_settings(credentials)
    missing_settings = missing_required_settings(credentials)
    settings_ready = not missing_settings and connection_error is None

header_col, action_col = st.columns([5, 1])

with header_col:
    st.markdown(
        """
<div class="app-kicker">Document retrieval workspace</div>
<h1 class="app-title">EAZI'S RAG Chatbot</h1>
<div class="app-subtitle">Search uploaded documents and answer from retrieved context.</div>
""",
        unsafe_allow_html=True,
    )

with action_col:
    st.write("")
    st.write("")
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    if settings_ready:
        render_status_card(
            "Credentials",
            "Ready",
            "Groq chat, OpenRouter embeddings, Supabase storage",
            "ready",
        )
    elif connection_error:
        render_status_card(
            "Credentials",
            "Connection issue",
            connection_error,
            "error",
        )
    else:
        render_status_card(
            "Credentials",
            "Incomplete",
            "Missing " + ", ".join(missing_settings),
            "warn",
        )

with status_col2:
    if st.session_state.doc_info:
        doc_info = st.session_state.doc_info
        render_status_card(
            "Document",
            doc_info["filename"],
            f"{doc_info['pages']} pages, {doc_info['chunks']} chunks",
            "ready",
        )
    else:
        render_status_card(
            "Document",
            "Not loaded",
            "Upload and process a file from the sidebar",
            "warn",
        )

with status_col3:
    render_status_card(
        "Retrieval",
        model,
        f"Top {top_k} chunks, sources {'on' if show_sources else 'off'}",
        "info",
    )

st.divider()

chat_title_col, chat_action_col = st.columns([5, 1])

with chat_title_col:
    st.markdown("### Chat")

with chat_action_col:
    st.caption(f"{len(st.session_state.messages)} messages")

if st.session_state.messages:
    for msg in st.session_state.messages:
        render_message(msg, show_sources)
else:
    if not settings_ready:
        empty_title = "Settings are required"
        empty_detail = "Open Settings in the sidebar and add the required credentials."
    elif not st.session_state.doc_info:
        empty_title = "No document loaded"
        empty_detail = "Upload and process a document from the sidebar."
    else:
        empty_title = "Ready for questions"
        empty_detail = "Ask about the uploaded document."

    st.markdown(
        f"""
<div class="chat-empty">
    <strong>{escape(empty_title)}</strong>
    {escape(empty_detail)}
</div>
""",
        unsafe_allow_html=True,
    )

prompt = st.chat_input(
    "Ask about the current document",
    disabled=not settings_ready or not st.session_state.doc_info,
)

if prompt:
    if supabase is None:
        st.warning("Complete settings before asking questions.")
    else:
        time_now = datetime.now().strftime("%I:%M %p")

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt,
                "time": time_now,
            }
        )

        with st.spinner("Searching document..."):
            try:
                query_embedding = get_embedding([prompt])[0]

                if hasattr(query_embedding, "tolist"):
                    query_embedding = query_embedding.tolist()

                result = supabase.rpc(
                    "match_documents",
                    {
                        "query_embedding": query_embedding,
                        "match_count": top_k,
                    },
                ).execute()

                if not result.data:
                    answer = "No relevant information found."
                    sources = []
                else:
                    context = "\n\n".join(
                        row["content"] for row in result.data
                    )

                    sources = [
                        row.get("metadata", {}).get(
                            "filename",
                            "Unknown source",
                        )
                        for row in result.data
                    ]

                    response = get_groq_client(
                        credentials
                    ).chat.completions.create(
                        model=model,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "Answer ONLY using the provided context. "
                                    "If the answer is not in the context, "
                                    "say you don't know."
                                ),
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"Context:\n{context}\n\n"
                                    f"Question: {prompt}"
                                ),
                            },
                        ],
                        temperature=0.2,
                    )

                    answer = response.choices[0].message.content

            except Exception as exc:
                answer = f"Error: {exc}"
                sources = []

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "time": datetime.now().strftime("%I:%M %p"),
            }
        )

        st.rerun()

st.divider()

st.caption("Built with Streamlit, Supabase, Groq, and OpenRouter embeddings")
