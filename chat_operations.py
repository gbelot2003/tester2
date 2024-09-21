from openai import OpenAI
from dotenv import load_dotenv
import os

# Inicializa la API de OpenAI
load_dotenv(override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chat_with_gpt(prompt, context=None):
    """Envía un prompt a ChatGPT junto con contexto vectorizado."""
    messages = []
    if context:
        messages.append({"role": "system", "content": context})
    
    messages.append({"role": "user", "content": prompt})
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=500,
        temperature=0.1
    )
    return response.choices[0].message.content
