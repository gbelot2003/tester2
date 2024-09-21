import chromadb

# Inicializar ChromaDB con SQLite
chroma_client = chromadb.PersistentClient(path="chromadb")

def store_chunks_in_chromadb(chunks, pdf_path, collection_name="pdf_collection"):
    """Vectoriza y almacena fragmentos de texto en ChromaDB."""
    collection = chroma_client.get_or_create_collection(collection_name)
    
    for i, (chunk, embedding) in enumerate(chunks):
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"{pdf_path}_chunk_{i}"]
        )
        print(f"Fragmento {i+1} almacenado en ChromaDB.")

def search_in_chromadb(query_embedding, collection_name="pdf_collection"):
    """Busca en ChromaDB utilizando el embedding de una consulta."""
    collection = chroma_client.get_collection(collection_name)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3  # número de resultados a devolver
    )
    if results['documents']:
        return results['documents']
    else:
        return "No se encontró información relevante."
