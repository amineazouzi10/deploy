from fastapi import FastAPI, Body
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

app = FastAPI()
# 1) Charger tokenizer et base model directement depuis HF Hub (sans l’avoir en local)
tokenizer = AutoTokenizer.from_pretrained("mistral‑7b‑instruct‑v0.2")
quant_config = BitsAndBytesConfig(load_in_8bit=True)  # ou config de quantification que vous souhaitez
model = AutoModelForCausalLM.from_pretrained(
    "mistral‑7b‑instruct‑v0.2",
    quantization_config=quant_config,
    device_map="auto"
)
# 2) Charger votre adapter depuis GCS
#     il faut que gcsfuse ou storage client soit dispo ; ici on monte le bucket dans /mnt/adapters
model.load_adapter("/mnt/adapters/adapter", config="pfeiffer", load_as="mistral_adapter")
model.set_adapter("mistral_adapter")

@app.post("/generate")
async def generate(prompt: str = Body(..., embed=True)):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=128)
    return {"text": tokenizer.decode(out[0], skip_special_tokens=True)}
