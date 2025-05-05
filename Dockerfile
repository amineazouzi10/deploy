FROM python:3.10-slim AS builder

# Installation des paquets nécessaires pour la phase de construction
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Installation des dépendances Python
RUN pip install --no-cache-dir \
    huggingface_hub \
    transformers

# Création du répertoire de cache
RUN mkdir -p /app/model_cache

# Définition des variables d'environnement pour HF
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV HF_HOME=/app/model_cache

# Pré-téléchargement du modèle et du tokenizer
# Note: Le token doit être passé lors du build avec --build-arg HF_TOKEN=your_token
ARG HF_TOKEN
# Vérifions que le token n'est pas vide
RUN if [ -z "$HF_TOKEN" ]; then echo "HF_TOKEN is required. Use --build-arg HF_TOKEN=your_token"; exit 1; fi

# Utilisons le token directement dans le script Python pour éviter les problèmes d'interpolation
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download( \
        repo_id='mistralai/Mistral-7B-Instruct-v0.2', \
        cache_dir='/app/model_cache', \
        token='$HF_TOKEN' \
    )"

# Image finale
FROM python:3.10-slim

# Installation des paquets et dépendances nécessaires
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Installation des dépendances Python
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    torch \
    transformers \
    google-cloud-storage \
    bitsandbytes \
    accelerate \
    safetensors \
    huggingface_hub

# Copie du code de l'application
COPY inference/ /app/
WORKDIR /app

# Copie du cache des modèles depuis l'étape de construction
COPY --from=builder /app/model_cache /app/model_cache

# Définition des variables d'environnement
ENV PORT=8080
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV HF_HOME=/app/model_cache

# Exposition du port
EXPOSE 8080

# Commande de démarrage
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--timeout-keep-alive", "120"]