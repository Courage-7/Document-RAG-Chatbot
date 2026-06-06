from openai import OpenAI

from app.config.settings import (
    OPENROUTER_BASE_URL,
    OPENROUTER_EMBEDDING_DIMENSIONS,
    OPENROUTER_EMBEDDING_MODEL,
    get_setting,
)


def get_openrouter_client(api_key: str | None = None) -> OpenAI:
    resolved_api_key = api_key or get_setting("OPENROUTER_API_KEY")

    if not resolved_api_key:
        raise ValueError("Missing OPENROUTER_API_KEY.")

    return OpenAI(
        api_key=resolved_api_key,
        base_url=OPENROUTER_BASE_URL,
    )


def get_embedding(texts, api_key: str | None = None):
    if isinstance(texts, str):
        texts = [texts]

    response = get_openrouter_client(api_key).embeddings.create(
        model=OPENROUTER_EMBEDDING_MODEL,
        input=texts,
        dimensions=OPENROUTER_EMBEDDING_DIMENSIONS,
    )

    return [item.embedding for item in response.data]
