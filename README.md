# Document RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with Python, Streamlit, Groq, OpenRouter-hosted Gemini embeddings, and Supabase vector search.

## Features

- Upload PDF, Word, and text documents
- Split uploaded documents into retrievable chunks
- Store embeddings in Supabase
- Ask questions about uploaded documents
- Generate answers with Groq chat models
- Enter required credentials from the Streamlit settings section
- Optionally save credentials to a local `.env` file

## Required Credentials

The Streamlit settings section requests these values:

- `GROQ_API_KEY` for chat completions
- `OPENROUTER_API_KEY` for Gemini embeddings through OpenRouter
- `SUPABASE_URL` for the Supabase project URL
- `SUPABASE_KEY` for the Supabase API key

Credentials entered in the UI are applied to the current Streamlit session. Select `Save to .env` in settings to persist them locally. The `.env` file is ignored by git.

## Installation

```bash
uv sync
```

## Run App

```bash
python -m streamlit run app/UI/Streamlit_app.py
```

## Tech Stack

- Python
- Streamlit
- Groq
- OpenRouter
- Supabase
- LangChain

## Author

Israel
