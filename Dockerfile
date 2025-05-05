FROM python:3.10-slim

# Installer les dépendances système minimales
RUN apt-get update && \
    apt-get install -y gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copier le code de l'application
COPY inference/ /app/
WORKDIR /app

# Installer les dépendances Python
RUN pip install --no-cache-dir \
    fastapi uvicorn torch transformers \
    google-cloud-storage bitsandbytes \
    huggingface_hub safetensors accelerate

# Variables d'environnement pour que FastAPI écoute sur le port défini
ENV PORT=8080
EXPOSE 8080

# Lancement du serveur
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--timeout-keep-alive", "120"]
