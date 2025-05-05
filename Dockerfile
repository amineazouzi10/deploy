FROM python:3.10-slim

# Installation des paquets et dépendances nécessaires
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Création du répertoire de cache
RUN mkdir -p /app/model_cache

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

# Définition des variables d'environnement
ENV PORT=8080
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV HF_HOME=/app/model_cache

# Exposition du port
EXPOSE 8080

# Commande de démarrage avec augmentation du délai de démarrage
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--timeout-keep-alive", "120"]