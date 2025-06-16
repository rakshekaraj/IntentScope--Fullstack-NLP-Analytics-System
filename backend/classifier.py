# backend/classifier.py

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Set paths to model and tokenizer
MODEL_PATH = "model/intent_model.pt"
TOKENIZER_PATH = "model/tokenizer"

# Labels for classification
labels = ['weather', 'reminder', 'music', 'general', 'ambiguous']  # Example labels

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = DistilBertTokenizer.from_pretrained(TOKENIZER_PATH)
model = DistilBertForSequenceClassification.from_pretrained(TOKENIZER_PATH, num_labels=len(labels))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Classify function
def classify_query(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
        return labels[pred_class], round(confidence, 3)


# This file wraps my trained PyTorch model and tokenizer so we can classify any user query into one of the predefined intents like weather, music, etc. 
# It's the core ML inference module.