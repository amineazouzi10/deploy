FROM python:3.10-slim
RUN pip install --no-cache-dir fastapi uvicorn torch transformers google-cloud-storage
# On monte le bucket GCS via Cloud Run ‘mounts’ ou via gcsfuse
COPY inference/ /app/
WORKDIR /app
# Lors du run, on montera gs://my‑mistral‑adapters dans /mnt/adapters
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
