FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    fastapi \
    uvicorn \
    transformers

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]