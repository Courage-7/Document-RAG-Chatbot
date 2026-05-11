import together

from app.config.settings import TOGETHER_API_KEY
from app.rag.retriever import retrieve_chunks

together.api_key = TOGETHER_API_KEY

def ask_chatbot(question):
    retrieved_docs = retrieve_chunks(question)

    context = ""

    for doc in retrieved_docs:
        context += doc["content"] + "\n"

    prompt = f"""
    Answer ONLY from the provided context.

    Context:
    {context}

    Question:
    {question}

    If the answer is not in the context, say:
    "I could not find the answer in the uploaded document."
    """

    response = together.Complete.create(
        prompt=prompt,
        model="mistralai/Mistral-7B-Instruct-v0.1",
        max_tokens=300
    )

    return response["output"]["choices"][0]["text"]