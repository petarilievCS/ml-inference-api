from transformers import pipeline

classifier = pipeline("sentiment-analysis")

print(classifier("I'm really enjoying this project!"))