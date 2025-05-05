from fastapi import FastAPI, Body, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
import logging
import time

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Chemin local pour stocker les modèles et les tokenizers
CACHE_DIR = "/app/model_cache"

# Configurer le modèle de base
BASE = "mistralai/Mistral-7B-Instruct-v0.2"
quant_config = BitsAndBytesConfig(load_in_8bit=True)

# Variables globales pour le modèle et le tokenizer
tokenizer = None
model = None


@app.on_event("startup")
async def startup_event():
    """Charge le modèle et le tokenizer au démarrage de l'application."""
    global tokenizer, model

    # Obtenir le token HF de l'environnement si défini
    hf_token = os.environ.get("HF_HUB_TOKEN", None)

    try:
        start_time = time.time()
        logger.info(f"Chargement du tokenizer depuis {BASE}")

        # Essayer d'abord avec local_files_only=True pour utiliser les fichiers pré-téléchargés
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                BASE,
                token=hf_token,
                cache_dir=CACHE_DIR,
                local_files_only=True
            )
        except Exception as e:
            logger.warning(f"Impossible de charger le tokenizer localement: {str(e)}. Essai en ligne...")
            tokenizer = AutoTokenizer.from_pretrained(
                BASE,
                token=hf_token,
                cache_dir=CACHE_DIR,
                local_files_only=False
            )

        logger.info(f"Chargement du modèle depuis {BASE}")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                BASE,
                token=hf_token,
                quantization_config=quant_config,
                device_map="auto",
                cache_dir=CACHE_DIR,
                local_files_only=True
            )
        except Exception as e:
            logger.warning(f"Impossible de charger le modèle localement: {str(e)}. Essai en ligne...")
            model = AutoModelForCausalLM.from_pretrained(
                BASE,
                token=hf_token,
                quantization_config=quant_config,
                device_map="auto",
                cache_dir=CACHE_DIR,
                local_files_only=False
            )

        # Charger l'adaptateur depuis le montage GCS s'il existe
        adapter_path = "/mnt/adapter"
        if os.path.exists(adapter_path) and os.listdir(adapter_path):  # Vérifier que le dossier n'est pas vide
            logger.info(f"Chargement de l'adaptateur depuis {adapter_path}")
            model.load_adapter(adapter_path, config="pfeiffer", load_as="mistral_adapter")
            model.set_adapter("mistral_adapter")
            logger.info("Adaptateur chargé avec succès")
        else:
            logger.warning(f"Le chemin d'adaptateur {adapter_path} n'existe pas ou est vide.")

        load_time = time.time() - start_time
        logger.info(f"Modèle chargé avec succès en {load_time:.2f} secondes")

    except Exception as e:
        logger.error(f"Erreur critique lors du chargement du modèle: {str(e)}")
        # Ne pas planter le serveur, mais signaler qu'il y a un problème


@app.get("/health")
async def health_check():
    """Endpoint pour les health checks."""
    if tokenizer is None or model is None:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas encore chargé")
    return {"status": "ok", "model": BASE}


@app.post("/generate")
async def generate(prompt: str = Body(..., embed=True)):
    """Endpoint pour générer du texte à partir d'un prompt."""
    global tokenizer, model

    if tokenizer is None or model is None:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas encore prêt")

    try:
        logger.info(f"Génération pour un prompt de {len(prompt)} caractères")

        # Formatage du prompt pour Mistral Instruct
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"

        # Tokenisation et génération
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

        gen_start_time = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        gen_time = time.time() - gen_start_time

        # Décodage de la sortie
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extraire uniquement la réponse (pas le prompt)
        response = response.split("[/INST]")[-1].strip()

        logger.info(f"Génération terminée en {gen_time:.2f} secondes")
        return {"text": response}

    except Exception as e:
        logger.error(f"Erreur lors de la génération: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))