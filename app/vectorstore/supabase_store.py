import os
from supabase import create_client
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# =========================
# Supabase Client
# =========================
supabase_client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# =========================
# OpenAI Client
# =========================
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# =========================
# Embeddings
# =========================
def get_embedding(texts):
    if isinstance(texts, str):
        texts = [texts]

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    return [item.embedding for item in response.data]


# =========================
# Retrieve (Vector Search)
# =========================
def retrieve_chunks(question, top_k=5):

    question_embedding = get_embedding(question)[0]

    response = supabase_client.rpc(
        "match_documents",
        {
            "query_embedding": question_embedding,
            "match_count": top_k
        }
    ).execute()

    return response.data


# =========================
# Store Embeddings
# =========================
def store_embeddings(chunks, embeddings, filename=None):

    data = []

    for chunk, embedding in zip(chunks, embeddings):

        data.append({
            "content": chunk,
            "embedding": embedding,
            "metadata": {
                "filename": filename
            }
        })

    supabase_client.table("documents").insert(data).execute()