FROM python:3.10-slim

# Installer les dépendances système minimales
RUN apt-get update && \
    apt-get install -y gcc g++ git && \
    rm -rf /var/lib/apt/lists/*

# Créer les répertoires nécessaires
RUN mkdir -p /app/model_cache
WORKDIR /app

# Copier les fichiers d'application
COPY *.py /app/

# Installer les dépendances Python
RUN pip install --no-cache-dir \
    fastapi uvicorn torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu \
    transformers==4.35.2 \
    google-cloud-storage \
    huggingface_hub \
    safetensors \
    accelerate \
    peft \
    sentencepiece \
    spacy \
    re

# Téléchargement du modèle spaCy français
RUN python -m spacy download fr_core_news_sm

# Variables d'environnement pour que FastAPI écoute sur le port défini
ENV PORT=8080
EXPOSE 8080

# Lancement du serveur avec un timeout plus long pour permettre
# le chargement du modèle et les longues inférences
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--timeout-keep-alive", "300"]