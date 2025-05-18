# Imagen base con Python 3.12
FROM python:3.12-slim

# Evitar creación de bytecode y hacer logging sin buffer
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Crear y usar el directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar toda la carpeta actual al contenedor
COPY . .

# Instalar pip actualizado y dependencias necesarias
# Primero llama_index desde código fuente en modo editable
RUN pip install --upgrade pip && \
    pip install -e llama_index && \
    pip install \
        llama-index-llms-ollama \
        llama-index-embeddings-huggingface

# Comando por defecto para iniciar la app
CMD ["python", "starter.py"]
