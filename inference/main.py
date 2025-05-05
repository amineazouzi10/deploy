from fastapi import FastAPI, Body, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import logging
import time
import re
import json
import spacy

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Chemin local pour stocker les modèles et les tokenizers
CACHE_DIR = "/app/model_cache"

# Configurer le modèle de base
BASE = "mistralai/Mistral-7B-Instruct-v0.2"

# Variables globales pour le modèle et le tokenizer
tokenizer = None
model = None
nlp = None  # pour spaCy

# Obtenir le token HF de l'environnement
HF_HUB_TOKEN = os.environ.get("HF_HUB_TOKEN", "")


@app.on_event("startup")
async def startup_event():
    """Charge le modèle et le tokenizer au démarrage de l'application."""
    global tokenizer, model, nlp

    try:
        start_time = time.time()

        # Charger le modèle spaCy pour le français
        try:
            logger.info("Chargement du modèle spaCy pour le français...")
            nlp = spacy.load("fr_core_news_sm")
            logger.info("Modèle spaCy chargé avec succès!")
        except Exception as e:
            logger.warning(f"Erreur lors du chargement du modèle spaCy: {e}")
            logger.info("Installation du modèle spaCy français...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "fr_core_news_sm"])
            try:
                nlp = spacy.load("fr_core_news_sm")
                logger.info("Modèle spaCy installé et chargé avec succès!")
            except Exception as e:
                logger.error(f"Impossible de charger le modèle spaCy: {e}")
                # Fallback à aucune lemmatisation
                nlp = None

        logger.info(f"Chargement du tokenizer depuis {BASE}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                BASE,
                cache_dir=CACHE_DIR,
                local_files_only=True,
                token=HF_HUB_TOKEN
            )
        except Exception as e:
            logger.warning(f"Impossible de charger le tokenizer localement: {e}. Essai en ligne...")
            tokenizer = AutoTokenizer.from_pretrained(
                BASE,
                cache_dir=CACHE_DIR,
                local_files_only=False,
                token=HF_HUB_TOKEN
            )

        logger.info(f"Chargement du modèle depuis {BASE}")
        try:
            # Pour CPU uniquement - sans quantification 8-bit qui nécessite CUDA
            model = AutoModelForCausalLM.from_pretrained(
                BASE,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                low_cpu_mem_usage=True,
                cache_dir=CACHE_DIR,
                local_files_only=True,
                token=HF_HUB_TOKEN
            )
        except Exception as e:
            logger.warning(f"Impossible de charger le modèle localement: {e}. Essai en ligne...")
            model = AutoModelForCausalLM.from_pretrained(
                BASE,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                low_cpu_mem_usage=True,
                cache_dir=CACHE_DIR,
                local_files_only=False,
                token=HF_HUB_TOKEN
            )

        # Charger l'adaptateur depuis le montage GCS s'il existe
        adapter_path = "/mnt/adapter"
        if os.path.isdir(adapter_path) and os.listdir(adapter_path):
            logger.info(f"Chargement de l'adaptateur depuis {adapter_path}")
            model.load_adapter(adapter_path, config="pfeiffer", load_as="mistral_adapter")
            model.set_adapter("mistral_adapter")
            logger.info("Adaptateur chargé avec succès")
        else:
            logger.warning(f"Le chemin d'adaptateur {adapter_path} n'existe pas ou est vide.")

        load_time = time.time() - start_time
        logger.info(f"Modèle chargé en {load_time:.2f} sec")

    except Exception as e:
        logger.error(f"Erreur critique lors du chargement du modèle: {e}")
        # Logguer plus de détails pour le débogage
        import traceback
        logger.error(traceback.format_exc())


# Fonctions de traitement de texte
def clean_text(text):
    """Nettoyage de base: mettre en minuscules et supprimer les espaces supplémentaires."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def lemmatize_text(text):
    """Lemmatiser le texte avec spaCy."""
    if not text or nlp is None:
        return text
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])


# Fonction pour construire le prompt
def build_prompt(project_info):
    """
    Construire un prompt à partir des caractéristiques du projet:
      - Nom du projet, Description, Durée (mois), Complexité (1-5), Secteur, Tâches Identifiées.
    Puis demander une sortie JSON de:
      - Compétences Requises, Employés Alloués, Répartition par Compétences.
    """
    nom = lemmatize_text(clean_text(project_info.get("Nom du projet", "")))
    description = lemmatize_text(clean_text(project_info.get("Description", "")))
    duree = project_info.get("Durée (mois)", "")
    complexite = project_info.get("Complexité (1-5)", "")
    secteur = clean_text(project_info.get("Secteur", ""))
    taches = project_info.get("Tâches Identifiées", "")

    prompt = (f"Nom du projet: {nom}\n"
              f"Description: {description}\n"
              f"Durée (mois): {duree}\n"
              f"Complexité (1-5): {complexite}\n"
              f"Secteur: {secteur}\n"
              f"Tâches Identifiées: {taches}\n\n"
              "### Instruction:\n"
              "Fournis les informations en format JSON pour:\n"
              "- Compétences Requises\n"
              "- Employés Alloués\n"
              "- Répartition par Compétences\n\n"
              "### Réponse:\n")
    return prompt


@app.get("/health")
async def health_check():
    """Endpoint pour les health checks."""
    if tokenizer is None or model is None:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas encore chargé")
    return {"status": "ok", "model": BASE}


@app.post("/generate")
async def generate(prompt: str = Body(..., embed=True)):
    """Endpoint pour générer du texte à partir d'un prompt standard."""
    global tokenizer, model

    if tokenizer is None or model is None:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas encore prêt")

    try:
        logger.info(f"Génération pour un prompt de {len(prompt)} car.")

        # Formatage du prompt pour Mistral Instruct
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"

        # Tokenisation et génération
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

        gen_start_time = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        gen_time = time.time() - gen_start_time

        # Décodage et extraction de la réponse
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("[/INST]")[-1].strip()

        logger.info(f"Génération terminée en {gen_time:.2f} sec")
        return {"text": response}

    except Exception as e:
        logger.error(f"Erreur lors de la génération: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/project")
async def generate_project_recommendations(project_info: dict = Body(...)):
    """Endpoint pour générer des recommandations pour un projet."""
    global tokenizer, model

    if tokenizer is None or model is None:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas encore prêt")

    try:
        # Construire le prompt spécifique au projet
        prompt = build_prompt(project_info)
        logger.info(f"Génération pour un projet: {project_info.get('Nom du projet', '')}")

        # Tokenisation et génération
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        gen_start_time = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=2000,
            do_sample=False,  # génération déterministe
            eos_token_id=tokenizer.eos_token_id
        )
        gen_time = time.time() - gen_start_time

        # Décodage de la sortie
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extraire uniquement la partie réponse (après "### Réponse:")
        if "### Réponse:" in generated_text:
            response = generated_text.split("### Réponse:")[-1].strip()
        else:
            response = generated_text.strip()

        # Tentative de correction des problèmes de formatage JSON
        response = response.strip()
        # Si la réponse commence par un backtick (courant dans les blocs de code markdown), le supprimer
        if response.startswith("```json"):
            response = response.split("```json", 1)[1]
        if response.startswith("```"):
            response = response.split("```", 1)[1]
        if response.endswith("```"):
            response = response.rsplit("```", 1)[0]

        # Essayer de parser la réponse comme JSON
        try:
            json_response = json.loads(response)
            logger.info(f"Génération terminée en {gen_time:.2f} sec - Réponse JSON valide")
            return json_response
        except json.JSONDecodeError as e:
            logger.warning(f"Réponse non JSON valide: {e}")
            # Retourner le texte brut si le parsing JSON échoue
            return {"text": response, "warning": "La réponse n'est pas un JSON valide"}

    except Exception as e:
        logger.error(f"Erreur lors de la génération: {e}")
        raise HTTPException(status_code=500, detail=str(e))