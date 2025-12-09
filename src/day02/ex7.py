#interactive

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np

# -------------------------
# CONFIG
# -------------------------
MODEL_DIR = "./fine_tuned_model"
MAX_LENGTH = 128

# -------------------------
# LOAD MODEL & TOKENIZER
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

print("Model and tokenizer loaded. Ready to test!")

# -------------------------
# INTERACTIVE PREDICTION
# -------------------------
predictions = []
true_labels = []

while True:
    text = input("\nEnter a text to classify (or 'exit' to quit):\n> ")
    if text.lower() in ["exit", "quit"]:
        break

    # Ask for optional true label
    label_input = input("Optional: enter the true label (0/1) or leave blank: ")
    true_label = int(label_input) if label_input.strip() != "" else None

    # Tokenize input
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        pred_label = int(torch.argmax(outputs.logits, dim=1).item())

    print(f"Predicted label: {pred_label}")

    predictions.append(pred_label)
    if true_label is not None:
        true_labels.append(true_label)

# -------------------------
# COMPUTE STATISTICS (IF LABELS PROVIDED)
# -------------------------
if true_labels:
    print("\n--- Evaluation Statistics ---")
    acc = accuracy_score(true_labels, predictions)
    prec = precision_score(true_labels, predictions, average="weighted")
    rec = recall_score(true_labels, predictions, average="weighted")
    f1 = f1_score(true_labels, predictions, average="weighted")
    cm = confusion_matrix(true_labels, predictions)
    report = classification_report(true_labels, predictions)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (weighted): {prec:.4f}")
    print(f"Recall (weighted): {rec:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

print("\n✅ Interactive testing session finished!")
