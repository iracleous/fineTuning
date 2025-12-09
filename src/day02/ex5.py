# This script:

# Loads your fine-tuned model from ./fine_tuned_model
# Loads your test dataset from ./local_data
# Computes predictions, accuracy, F1, confusion matrix
# Saves evaluation_results.json for plotting



import json
import csv
from pathlib import Path
import torch

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# -------------------------
# CONFIG
# -------------------------
MODEL_DIR = "./fine_tuned_model"
TEST_DATA_DIR = "./local_data"
TEST_FILE = "test2.jsonl"
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

MAX_LENGTH = 128
MAX_TEST_SAMPLES = 20

OUTPUT_METRICS_JSON = "evaluations/evaluation_metrics2.json"
OUTPUT_PREDICTIONS_CSV = "evaluations/predictions2.csv"

# -------------------------
# LOAD MODEL & TOKENIZER
# -------------------------
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# -------------------------
# LOAD TEST DATASET
# -------------------------
print("Loading test dataset...")
test_dataset = load_dataset("json", data_files=str(Path(TEST_DATA_DIR) / TEST_FILE))["train"]

# Restrict to N samples
test_dataset = test_dataset.select(range(min(MAX_TEST_SAMPLES, len(test_dataset))))

# -------------------------
# TOKENIZE TEST DATA
# -------------------------
def tokenize(example):
    return tokenizer(
        example[TEXT_COLUMN],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

tokenized = test_dataset.map(tokenize, batched=True)
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", LABEL_COLUMN])

# -------------------------
# RUN PREDICTIONS
# -------------------------
print("Running predictions...")
all_preds = []
all_labels = []

for batch in tokenized:
    input_ids = batch["input_ids"].unsqueeze(0)
    attention_mask = batch["attention_mask"].unsqueeze(0)
    label = int(batch[LABEL_COLUMN])  # ensure plain int

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        pred = torch.argmax(outputs.logits, dim=1).item()

    all_preds.append(pred)
    all_labels.append(label)

# -------------------------
# SAVE PREDICTIONS TO FILE
# -------------------------
with open(OUTPUT_PREDICTIONS_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["label", "prediction"])
    for lbl, pred in zip(all_labels, all_preds):
        writer.writerow([lbl, pred])

print(f"Saved predictions to {OUTPUT_PREDICTIONS_CSV}")

# -------------------------
# SKLEARN METRICS
# -------------------------
metrics = {
    "accuracy": accuracy_score(all_labels, all_preds),
    "precision_weighted": precision_score(all_labels, all_preds, average="weighted"),
    "recall_weighted": recall_score(all_labels, all_preds, average="weighted"),
    "f1_weighted": f1_score(all_labels, all_preds, average="weighted"),
    "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
    "classification_report": classification_report(all_labels, all_preds, output_dict=True)
}

with open(OUTPUT_METRICS_JSON, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Saved evaluation metrics to {OUTPUT_METRICS_JSON}")

print("\n✅ Done! (No visualizations)")
