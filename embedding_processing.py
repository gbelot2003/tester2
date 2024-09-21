from openai import OpenAI
from dotenv import load_dotenv
import os

# Inicializa la API de OpenAI
load_dotenv(override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding_for_chunk(chunk):
    """Obtiene el embedding de un fragmento de texto utilizando OpenAI."""
    response = client.embeddings.create(
        input=chunk,
        model="text-embedding-ada-002"
    )
    embedding = response.data[0].embedding
    return embedding
