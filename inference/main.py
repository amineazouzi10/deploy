from fastapi import FastAPI, Body
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Chemin local pour stocker les modèles et les tokenizers
CACHE_DIR = "/app/model_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# 1) Charger le tokenizer et le modèle de base directement depuis HF Hub avec mise en cache
BASE = "mistralai/Mistral-7B-Instruct-v0.2"
quant_config = BitsAndBytesConfig(load_in_8bit=True)

# Obtenir le token HF de l'environnement si défini
hf_token = "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"

try:
    logger.info(f"Chargement du tokenizer depuis {BASE}")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE,
        token=hf_token,
        cache_dir=CACHE_DIR,
        local_files_only=False  # Essayer d'abord en ligne, puis utiliser le cache
    )

    logger.info(f"Chargement du modèle depuis {BASE}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        token=hf_token,
        quantization_config=quant_config,
        device_map="auto",
        cache_dir=CACHE_DIR,
        local_files_only=False  # Essayer d'abord en ligne, puis utiliser le cache
    )

    # 2) Charger l'adaptateur depuis le montage GCS
    adapter_path = "/mnt/adapter"
    if os.path.exists(adapter_path):
        logger.info(f"Chargement de l'adaptateur depuis {adapter_path}")
        model.load_adapter(adapter_path, config="pfeiffer", load_as="mistral_adapter")
        model.set_adapter("mistral_adapter")
    else:
        logger.warning(f"Le chemin d'adaptateur {adapter_path} n'existe pas.")

except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
    # Ne pas planter le serveur, mais signaler qu'il y a un problème
    # Dans un environnement de production, vous pourriez vouloir réessayer ou utiliser
    # un modèle de secours


@app.get("/health")
async def health_check():
    """Endpoint pour les health checks."""
    return {"status": "ok"}


@app.post("/generate")
async def generate(prompt: str = Body(..., embed=True)):
    try:
        logger.info(f"Génération pour un prompt de {len(prompt)} caractères")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=128)
        response = tokenizer.decode(out[0], skip_special_tokens=True)
        return {"text": response}
    except Exception as e:
        logger.error(f"Erreur lors de la génération: {str(e)}")
        return {"error": str(e)}, 500