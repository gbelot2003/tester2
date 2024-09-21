from pdf_processing import extract_text_from_pdf, split_text_into_chunks
from embedding_processing import get_embedding_for_chunk
from chromadb_operations import store_chunks_in_chromadb, search_in_chromadb
from chat_operations import chat_with_gpt

def process_multiple_pdfs(pdf_paths):
    """Procesa y almacena varios archivos PDF."""
    for pdf_path in pdf_paths:
        # Extraemos y fragmentamos cada PDF
        text = extract_text_from_pdf(pdf_path)
        chunks = split_text_into_chunks(text)
        
        # Obtenemos embeddings para cada fragmento y los almacenamos en ChromaDB
        chunks_with_embeddings = [(chunk, get_embedding_for_chunk(chunk)) for chunk in chunks]
        store_chunks_in_chromadb(chunks_with_embeddings, pdf_path)

        print(f"PDF {pdf_path} completamente almacenado en ChromaDB.")

def main():
    # Lista de rutas de archivos PDF
    pdf_paths = [
        "files/encomiendas.pdf",
    ]
    
    # Procesamos todos los PDFs
    process_multiple_pdfs(pdf_paths)
    
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
