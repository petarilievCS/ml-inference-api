from transformers import pipeline

# Load pretrained pipeline
classifier = pipeline("sentiment-analysis")

# Save the model and tokenizer locally
classifier.model.save_pretrained("./saved_model")
classifier.tokenizer.save_pretrained("./saved_model")