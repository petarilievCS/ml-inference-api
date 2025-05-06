from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()

model_path = "./saved_model"
classifier = pipeline("sentiment-analysis",
                       model=AutoModelForSequenceClassification.from_pretrained(model_path),
                       tokenizer=AutoTokenizer.from_pretrained(model_path))

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