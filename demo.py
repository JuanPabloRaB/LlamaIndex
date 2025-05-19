from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.workflow import Context, JsonPickleSerializer
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import asyncio
import os
import tempfile
import json

# Configurar primero
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)

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

# Herramientas
def multiply(a: float, b: float) -> float:
    return a * b

async def search_documents(query: str) -> str:
    response = await query_engine.aquery(query)
    return str(response)

# Herramienta que usa estado
async def set_name(ctx: Context, name: str) -> str:
    state = await ctx.get("state")
    state["name"] = name
    await ctx.set("state", state)
    return f"✅ Nombre guardado: {name}"

# Crear agente
agent = AgentWorkflow.from_tools_or_functions(
    [multiply, search_documents, set_name],
    llm=Settings.llm,
    system_prompt="""Eres un asistente útil que puede realizar 
    cálculos, recordar datos del usuario y buscar en documentos 
    para responder preguntas, sin utilizar internet. Decide cuál herramienta usar en cada caso.""",
    initial_state={"name": "desconocido"},
)

# Archivo de persistencia del contexto
ctx_path = "estado_contexto.json"
serializer = JsonPickleSerializer()

# Cargar o crear contexto
if os.path.exists(ctx_path):
    print("🔄 Restaurando contexto anterior...")
    with open(ctx_path, "r") as f:
        ctx_data = json.load(f)
    ctx = Context.from_dict(agent, ctx_data, serializer=serializer)
else:
    ctx = Context(agent)
    asyncio.run(ctx.set("history", []))
    asyncio.run(ctx.set("name", "desconocido"))

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
            # Agregar al historial
            history = await ctx.get("history") or []
            history.append({
                "query": user_input,
                "response": str(response)
            })
            await ctx.set("history", history)

        except Exception as e:
            print(f"Ocurrió un error: {e}")

    print("💾 Guardando contexto en disco...")
    ctx_dict = ctx.to_dict(serializer=serializer)
    with open(ctx_path, "w") as f:
        json.dump(ctx_dict, f)

    # print("\n🧠 Historial de la conversación:")
    # history = await ctx.get("history") or []
    # for i, item in enumerate(history, 1):
    #     print(f"{i}. Q: {item['query']}\n   A: {item['response']}\n")

if __name__ == "__main__":
    asyncio.run(main())
    temp_dir.cleanup()
