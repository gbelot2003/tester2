from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import fitz  # PyMuPDF
import os
import chromadb

# Inicializa la API de OpenAI
load_dotenv(override=True)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Inicializar ChromaDB con SQLite
chroma_client = chromadb.PersistentClient(path="chromadb")

def extract_text_from_pdf(pdf_path):
    """Extrae el texto de un archivo PDF."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def split_text_into_chunks(text, max_tokens=100):
    """Divide el texto en fragmentos más pequeños para evitar los límites de la API."""
    words = text.split()
    chunks = []
    current_chunk = []
    
    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    
    # Añadir el último fragmento si queda algo
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def get_embedding_for_chunk(chunk):
    """Obtiene el embedding de un fragmento de texto."""
    response = client.embeddings.create(
        input=chunk,
        model="text-embedding-ada-002"
    )
    embedding = response.data[0].embedding
    return embedding

def store_pdf_to_chromadb(pdf_path, collection_name="pdf_collection"):
    """Vectoriza y almacena fragmentos de un PDF en ChromaDB."""
    text = extract_text_from_pdf(pdf_path)  # Extraemos el texto del PDF
    chunks = split_text_into_chunks(text)   # Dividimos el texto en fragmentos
    
    # Crea o conecta a una colección en ChromaDB
    collection = chroma_client.get_or_create_collection(collection_name)
    
    for i, chunk in enumerate(chunks):
        embedding = get_embedding_for_chunk(chunk)  # Generamos el embedding para cada fragmento
        
        # Añade cada fragmento y su embedding a la colección como un documento independiente
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"{os.path.basename(pdf_path)}_chunk_{i}"]  # Usamos el nombre del archivo + número de fragmento
        )
        print(f"Fragmento {i+1} del PDF {pdf_path} almacenado en ChromaDB.")
    
    print(f"PDF {pdf_path} completamente almacenado en ChromaDB.")

def search_in_chromadb(query, collection_name="pdf_collection"):
    """Busca en ChromaDB utilizando el embedding de una consulta."""
    # Crea o conecta a la colección
    collection = chroma_client.get_collection(collection_name)

    # Obtén el embedding de la consulta
    query_embedding = get_embedding_for_chunk(query)

    # Realiza la búsqueda en ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3  # número de resultados a devolver
    )

    # Devuelve los documentos más cercanos al embedding de la consulta
    if results['documents']:
        return results['documents']  # Retorna los documentos relevantes
    else:
        return "No se encontró información relevante."

def chat_with_gpt(prompt, context=None):
    """Envía un prompt a ChatGPT junto con contexto vectorizado."""
    messages = []
    
    # Si el contexto es None o no es una cadena, se establece como una cadena vacía
    if not isinstance(context, str):
        context = ""

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
    
    # Almacenamos el contenido vectorizado del PDF en ChromaDB (fragmentado)
    store_pdf_to_chromadb(pdf_path)
    
    print("Conversación con ChatGPT (escribe 'salir' para terminar):")
    
    while True:
        user_input = input("Tú: ")
        if user_input.lower() == "salir":
            break

        # Recuperamos información relevante de la base de datos usando el input del usuario
        context = search_in_chromadb(user_input)

        # Si hay resultados relevantes, aseguramos que cada uno sea una cadena antes de concatenarlos
        if isinstance(context, list):
            context = " ".join([doc if isinstance(doc, str) else " ".join(doc) for doc in context])

        # Realizamos el chat con el contexto
        response = chat_with_gpt(user_input, context)
        print(f"ChatGPT: {response}")

if __name__ == "__main__":
    main()
