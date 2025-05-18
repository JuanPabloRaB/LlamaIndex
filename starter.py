from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
    Document
)
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.workflow import Context, JsonPickleSerializer
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import asyncio
import os
import json
import glob
import shutil

# Configurar modelos globales
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

Settings.llm = Ollama(
    model="llama3.2",
    base_url=ollama_url,
    request_timeout=360.0
)

# Función para cargar un JSON como documento
def json_to_document(path: str) -> Document:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    text = json.dumps(data, indent=2, ensure_ascii=False)
    return Document(text=text)


# --- Bloque para cargar o crear índice persistente ---
persist_dir = "storage"
docstore_file = os.path.join(persist_dir, "docstore.json")

# Verificar si docstore.json está corrupto o vacío para eliminar y regenerar
if os.path.exists(persist_dir) and os.path.exists(docstore_file):
    try:
        with open(docstore_file, "r", encoding="utf-8") as f:
            json.load(f)
    except json.JSONDecodeError:
        print("❌ El archivo docstore.json está corrupto o vacío. Se eliminará el directorio storage para regenerarlo.")
        shutil.rmtree(persist_dir)

if os.path.exists(persist_dir) and os.path.exists(docstore_file):
    # Cargar índice desde persistencia
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context, embed_model=Settings.embed_model)
else:
    # Crear índice desde documentos JSON y persistirlo
    if not os.path.exists("./data") or not os.listdir("./data"):
        print("⚠️ La carpeta ./data está vacía o no existe.")
        exit(1)

    documents = []
    for filepath in glob.glob("./data/*.json"):
        documents.append(json_to_document(filepath))

    # Agregar documentos cargados con SimpleDirectoryReader (por si hay otros formatos)
    documents += SimpleDirectoryReader(input_dir="./data").load_data()

    index = VectorStoreIndex.from_documents(documents=documents, embed_model=Settings.embed_model)
    index.storage_context.persist(persist_dir)
# --- Fin bloque de índice ---

# Crear motor de consulta
query_engine = index.as_query_engine(llm=Settings.llm)


# Herramientas
def multiply(a: float, b: float) -> float:
    """Multiplica dos números."""
    return a * b

async def search_documents(query: str) -> str:
    """Responde preguntas sobre documentos locales."""
    response = await query_engine.aquery(query)
    return str(response)

# Crear flujo del agente
agent = AgentWorkflow.from_tools_or_functions(
    [multiply, search_documents],
    llm=Settings.llm,
    system_prompt="""Eres un asistente útil que puede realizar cálculos y buscar en documentos para responder preguntas, sin utilizar internet ni información en línea. Decide cuál herramienta usar en cada caso.""",
)

# Ruta del archivo para persistencia del contexto
ctx_path = "ctx.json"

# Cargar o crear nuevo contexto
if os.path.exists(ctx_path):
    print("🔄 Restaurando contexto anterior...")
    ctx = Context.load(agent, ctx_path, serializer=JsonPickleSerializer())
else:
    ctx = Context(agent)
    ctx.state["history"] = []

# Función principal
async def main():
    print("Asistente listo. Escribe tu pregunta o 'salir' para terminar.\n")
    while True:
        user_input = input("¿Qué quieres? > ")
        if user_input.lower() == "salir":
            break
        try:
            response = await agent.run(user_input, ctx=ctx)
            print(str(response))

            # Guardar en el historial
            ctx.state.setdefault("history", []).append({
                "query": user_input,
                "response": str(response),
            })
        except Exception as e:
            print(f"Ocurrió un error: {e}")

    # Guardar contexto en disco
    print("💾 Guardando estado...")
    ctx.save(ctx_path, serializer=JsonPickleSerializer())

    print("\n🧠 Historial de la conversación:")
    for i, item in enumerate(ctx.state["history"], 1):
        print(f"{i}. Q: {item['query']} \n   A: {item['response']}\n")

# Ejecutar
if __name__ == "__main__":
    asyncio.run(main())
