from fastapi import FastAPI, Body
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os

app = FastAPI()

# 1) Load tokenizer and base model directly from HF Hub
quant_config = BitsAndBytesConfig(load_in_8bit=True)
BASE = "mistralai/Mistral-7B-Instruct-v0.2"  # Corrected model path with proper ASCII hyphens

# Get HF token from environment if set
hf_token = os.environ.get("HF_HUB_TOKEN", None)

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(BASE, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    token=hf_token,
    quantization_config=quant_config,
    device_map="auto"
)

# 2) Load your adapter from GCS mount
model.load_adapter("/mnt/adapter", config="pfeiffer", load_as="mistral_adapter")
model.set_adapter("mistral_adapter")

@app.post("/generate")
async def generate(prompt: str = Body(..., embed=True)):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=128)
    return {"text": tokenizer.decode(out[0], skip_special_tokens=True)}