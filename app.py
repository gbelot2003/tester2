from pdf_processing import extract_text_from_pdf, split_text_into_chunks
from embedding_processing import get_embedding_for_chunk
from chromadb_operations import store_chunks_in_chromadb, search_in_chromadb
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
        max_tokens=150,
        temperature=0.1
    )
    return response.choices[0].message.content

def main():
    # Ruta del archivo PDF
    pdf_path = "files/encomiendas.pdf"
    
    # Extraemos y fragmentamos el PDF
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text_into_chunks(text)
    
    # Obtenemos embeddings para cada fragmento y lo almacenamos en ChromaDB
    chunks_with_embeddings = [(chunk, get_embedding_for_chunk(chunk)) for chunk in chunks]
    store_chunks_in_chromadb(chunks_with_embeddings, pdf_path)
    
    print("Conversación con ChatGPT (escribe 'salir' para terminar):")
    
    while True:
        user_input = input("Tú: ")
        if user_input.lower() == "salir":
            break

        # Vectorizamos la consulta del usuario
        query_embedding = get_embedding_for_chunk(user_input)

        # Buscamos en ChromaDB
        context = search_in_chromadb(query_embedding)

        # Unimos los resultados relevantes
        if isinstance(context, list):
            flattened_context = []
            for doc in context:
                if isinstance(doc, list):
                    flattened_context.extend(doc)  # Si es una lista, la añadimos aplanada
                else:
                    flattened_context.append(doc)  # Si es un string, lo añadimos directamente
            
            context = " ".join(flattened_context)  # Concatenamos el contexto
        
        # Realizamos el chat con el contexto
        response = chat_with_gpt(user_input, context)
        print(f"ChatGPT: {response}")

if __name__ == "__main__":
    main()
