import os
import together
from dotenv import load_dotenv
from app.vectorstore.supabase_store import retrieve_chunks


#load the variables from your .env file
load_dotenv()

#Get the key directly
TOGETHER_API_KEY = os.getenv("Together_API_KEY")
together.api_key = TOGETHER_API_KEY


def ask_chatbot(question):
    retrieved_docx = retrieve_chunks(question)

    context = ""

    for doc in retrieved_docx:
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