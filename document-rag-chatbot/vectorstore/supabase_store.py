from supabase import create_client
from config import SUPABASE_URL, SUPABASE_KEY

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def store_embeddings(chunks, embeddings):
    data = []

    for chunk, embedding in zip(chunks, embeddings):
        data.append({
            "content": chunk,
            "embedding": embedding.tolist()
        })

    supabase.table("documents").insert(data).execute()