from pdf_processing import extract_text_from_pdf
from embedding_processing import get_embedding_for_chunk
from chromadb_operations import store_chunks_in_chromadb, search_in_chromadb
from chat_operations import chat_with_gpt
import traceback

def procesar_precios_con_chatgpt(texto_pdf):
    """Envía fragmentos de texto del PDF a ChatGPT para que procese los precios."""
    # Dividimos el texto del PDF en fragmentos manejables (por ejemplo, 1000 caracteres cada uno)
    fragmentos = [texto_pdf[i:i+1000] for i in range(0, len(texto_pdf), 1000)]
    
    precios = {}

    # Enviar cada fragmento a ChatGPT y pedir que encuentre los precios
    for fragmento in fragmentos:
        prompt = f"Encuentra los productos y sus precios en el siguiente texto:\n\n{fragmento}"
        try:
            # Llamamos a ChatGPT con el fragmento
            respuesta = chat_with_gpt(prompt)
            print(f"ChatGPT procesó el fragmento:\n{respuesta}")

            # Aquí podrías procesar la respuesta si ChatGPT devuelve los datos en un formato estructurado
            # o simplemente agregar lo que ChatGPT encontró directamente a la cotización.
            precios.update(parsear_respuesta_chatgpt(respuesta))  # Función para procesar la respuesta de ChatGPT

        except Exception as e:
            print(f"Error procesando el fragmento con ChatGPT: {e}")
            print(traceback.format_exc())
    
    return precios

def parsear_respuesta_chatgpt(respuesta):
    """Parsea la respuesta de ChatGPT para extraer los productos y precios."""
    precios = {}
    for line in respuesta.split("\n"):
        # Supongamos que ChatGPT devuelve el formato "Producto: $Precio"
        if ":" in line:
            partes = line.split(":")
            if len(partes) == 2:
                producto = partes[0].strip()
                precio = partes[1].strip()
                precios[producto] = precio
    return precios

def buscar_precios_en_pdf_con_chatgpt(pdf_path):
    """Extrae texto del PDF y delega la búsqueda de precios a ChatGPT."""
    try:
        # Extraer texto completo del PDF
        texto_pdf = extract_text_from_pdf(pdf_path)
        print(f"Texto extraído del PDF {pdf_path}: {texto_pdf[:200]}...")  # Verificar los primeros 200 caracteres

        # Procesar los fragmentos con ChatGPT para extraer precios
        precios = procesar_precios_con_chatgpt(texto_pdf)

        if not precios:
            print(f"No se encontraron precios válidos en el PDF {pdf_path}.")
        else:
            print(f"Precios encontrados en el PDF {pdf_path}: {precios}")
        return precios

    except Exception as e:
        print(f"Error al procesar el PDF {pdf_path}: {e}")
        print(traceback.format_exc())
        return {}

def solicitar_informacion_usuario(nombre=None, correo=None, telefono=None):
    """Solicita la información opcional del usuario solo si no está ya presente."""
    if not nombre:
        nombre = input("Por favor, proporciona tu nombre (requerido): ")
        if not nombre:
            nombre = "Cliente Anónimo"  # Si no proporciona un nombre, usamos un valor por defecto.

    if not correo:
        correo = input("Por favor, proporciona tu correo electrónico (opcional): ")

    if not telefono:
        telefono = input("Por favor, proporciona tu número de teléfono (opcional): ")

    return nombre, correo, telefono

def enviar_cotizacion_al_usuario(precios, nombre, correo=None, telefono=None):
    """Formatea y envía la cotización al usuario, incluyendo su información opcional."""
    cotizacion = f"Aquí está la cotización de precios para {nombre}:\n\n"
    for producto, precio in precios.items():
        cotizacion += f"{producto}: {precio}\n"

    cotizacion += "\nInformación del contacto:\n"
    cotizacion += f"Nombre: {nombre}\n"
    if correo:
        cotizacion += f"Correo: {correo}\n"
    if telefono:
        cotizacion += f"Teléfono: {telefono}\n"

    return cotizacion

def main():
    # Lista de rutas de archivos PDF
    pdf_paths = [
        "files/encomiendas.pdf",
    ]

    # Procesamos todos los PDFs
    print("Procesando PDFs...")
    for pdf_path in pdf_paths:
        store_chunks_in_chromadb(pdf_path)  # Puedes modificar esta función para almacenar fragmentos.

    # Almacenamos la información del usuario
    nombre = None
    correo = None
    telefono = None

    print("Conversación con ChatGPT (escribe 'salir' para terminar):")

    while True:
        user_input = input("Tú: ")

        if user_input.lower() == "salir":
            break

        # Detectar si el usuario solicita una cotización
        if "cotización" in user_input.lower() or "precios" in user_input.lower():
            print("Generando cotización de precios...")

            try:
                # Solo pedir la información del usuario si aún no la tenemos
                nombre, correo, telefono = solicitar_informacion_usuario(nombre, correo, telefono)

                # Buscar precios en uno o varios PDFs usando ChatGPT
                precios_totales = {}
                for pdf_path in pdf_paths:
                    precios = buscar_precios_en_pdf_con_chatgpt(pdf_path)
                    if not precios:
                        print(f"No se encontraron precios en el PDF {pdf_path}")
                    precios_totales.update(precios)

                if not precios_totales:
                    print("No se encontraron precios en los PDFs.")
                    continue

                # Formatear y enviar la cotización al usuario con su información
                cotizacion = enviar_cotizacion_al_usuario(
                    precios_totales, nombre, correo, telefono
                )
                print(cotizacion)

            except Exception as e:
                print(f"Error generando la cotización: {e}")
                print(traceback.format_exc())
            continue  # Pasamos a la siguiente iteración sin usar ChatGPT en este caso

        else:
            # Si no se solicita cotización, invocamos ChatGPT para responder al usuario
            try:
                query_embedding = get_embedding_for_chunk(user_input)

                # Buscamos en ChromaDB para contexto relevante
                context = search_in_chromadb(query_embedding)

                # Unimos los resultados relevantes
                if isinstance(context, list):
                    flattened_context = []
                    for doc in context:
                        if isinstance(doc, list):
                            flattened_context.extend(doc)
                        else:
                            flattened_context.append(doc)

                    context = " ".join(flattened_context)

                # Realizamos el chat con el contexto y la respuesta de GPT
                response = chat_with_gpt(user_input, context)
                print(f"ChatGPT: {response}")

            except Exception as e:
                print(f"Error procesando la consulta del usuario: {e}")
                print(traceback.format_exc())

if __name__ == "__main__":
    main()
