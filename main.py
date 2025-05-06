from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
classifier = pipeline("sentiment-analysis")

class TextInput(BaseModel):
    text: str

@app.post("/predict")
async def read_root(input: TextInput):
    output = classifier(input.text)
    return {
        "label": output["label"],
        "score": round(output["score"], 4)
    }