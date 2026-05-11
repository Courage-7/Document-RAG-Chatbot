from sentence_transformers import SentenceTransformer
from supabase import create_client
from app.config.settings import SUPABASE_URL, SUPABASE_KEY

model = SentenceTransformer("all-MiniLM-L6-v2")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def retrieve_chunks(question):
    question_embedding = model.encode(question).tolist()

    response = supabase.rpc(
        "match_documents",
        {
            "query_embedding": question_embedding,
            "match_threshold": 0.5,
            "match_count": 5
        }
    ).execute()

    return response.data