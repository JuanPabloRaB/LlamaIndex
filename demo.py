from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import asyncio
import os
import tempfile


# Configurar primero
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)  # Reemplaza 'llama3.2' si no existe


# Crear carpeta temporal
temp_dir = tempfile.TemporaryDirectory()
persist_dir = temp_dir.name

# Verifica si el índice ya fue guardado
index_path = os.path.join(persist_dir, "index_store.json")
if os.path.exists(persist_dir) and len(os.listdir(persist_dir)) > 0:
    print("Recuperando índice temporal desde disco...")
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
else:
    print("Creando índice temporal desde documentos...")
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=persist_dir)

query_engine = index.as_query_engine()


def multiply(a: float, b: float) -> float:
    return a * b


async def search_documents(query: str) -> str:
    response = await query_engine.aquery(query)
    return response  # 👈 Ahora retorna el objeto completo


agent = AgentWorkflow.from_tools_or_functions(
    [multiply, search_documents],
    llm=Settings.llm,
    system_prompt="""Eres un asistente útil que puede realizar 
    cálculos y buscar en documentos para responder preguntas, 
    sin utilizar internet ni información en línea. Decide cuál herramienta usar en cada caso.""",
)


async def main():
    print("Asistente listo. Escribe tu pregunta o 'salir' para terminar.\n")
    while True:
        user_input = input("¿Qué quieres? > ")
        if user_input.lower() == "salir":
            break
        try:
            response = await agent.run(user_input)
            print(str(response))

        except Exception as e:
            print(f"Ocurrió un error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
    temp_dir.cleanup()
