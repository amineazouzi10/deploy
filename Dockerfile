FROM python:3.10-slim

# Install necessary packages and dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    torch \
    transformers \
    google-cloud-storage \
    bitsandbytes \
    accelerate \
    safetensors

# Copy application code
COPY inference/ /app/
WORKDIR /app

# Set environment variables
ENV PORT=8080

# Expose the port
EXPOSE 8080

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]