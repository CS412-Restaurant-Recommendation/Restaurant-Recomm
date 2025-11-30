import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def predict_sentiment(text: str) -> int:
    """
    Predict a star rating (1–5) for a given review text.
    """
    tokens = tokenizer.encode(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        output = model(tokens)

    scores = output.logits.detach().numpy()[0]
    prediction = int(np.argmax(scores)) + 1  # model outputs 0–4 → convert to 1–5
    return prediction