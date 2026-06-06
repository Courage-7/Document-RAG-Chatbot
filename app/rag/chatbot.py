from openai import OpenAI

from app.config.settings import (
    DEFAULT_GROQ_MODEL,
    GROQ_BASE_URL,
    get_setting,
)
from app.vectorstore.supabase_store import retrieve_chunks


def get_groq_client(api_key: str | None = None) -> OpenAI:
    resolved_api_key = api_key or get_setting("GROQ_API_KEY")

    if not resolved_api_key:
        raise ValueError("Missing GROQ_API_KEY.")

    return OpenAI(
        api_key=resolved_api_key,
        base_url=GROQ_BASE_URL,
    )


def ask_chatbot(question, top_k=5, model=DEFAULT_GROQ_MODEL):
    retrieved_docs = retrieve_chunks(question, top_k=top_k)
    context = "\n".join(doc["content"] for doc in retrieved_docs)

    response = get_groq_client().chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer ONLY from the provided context. "
                    "If the answer is not in the context, say: "
                    "I could not find the answer in the uploaded document."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}",
            },
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content
