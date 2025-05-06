from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()
classifier = pipeline("sentiment-analysis")

class TextInput(BaseModel):
    text: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def read_root(input: TextInput):
    logging.info(f"Received input: {input.text}")
    output = classifier(input.text)[0]
    logging.info(f"Prediction: {output}")
    return {
        "label": output["label"],
        "score": round(output["score"], 4)
    }