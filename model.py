from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import time

model_path = "./saved_model"
classifier = pipeline("sentiment-analysis",
                       model=AutoModelForSequenceClassification.from_pretrained(model_path),
                       tokenizer=AutoTokenizer.from_pretrained(model_path))

def classify(prompt: str) -> dict:
    return classifier(prompt)[0]