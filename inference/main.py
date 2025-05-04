from fastapi import FastAPI, Body
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

app = FastAPI()
# 1) Charger tokenizer et base model directement depuis HF Hub (sans l'avoir en local)
quant_config = BitsAndBytesConfig(load_in_8bit=True)  # ou config de quantification que vous souhaitez
BASE = "mistralai/Mistral-7B-Instruct-v0.2"  # Corrected model path with proper hyphens
# pas besoin d'appel explicite à `login()` si HF_HUB_TOKEN est défini
tokenizer = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    use_auth_token=True,       # parfois nécessaire selon version
    quantization_config=quant_config,
    device_map="auto"
)
# 2) Charger votre adapter depuis GCS
#     il faut que gcsfuse ou storage client soit dispo ; ici on monte le bucket dans /mnt/adapters
model.load_adapter("/mnt/adapter", config="pfeiffer", load_as="mistral_adapter")  # Updated path to match volume mount
model.set_adapter("mistral_adapter")

@app.post("/generate")
async def generate(prompt: str = Body(..., embed=True)):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=128)
    return {"text": tokenizer.decode(out[0], skip_special_tokens=True)}