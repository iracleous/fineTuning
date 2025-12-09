# This script:

# Loads evaluation_results.json

# Creates matplotlib figures:
# ✔ Accuracy bar
# ✔ F1 bar
# ✔ Confusion matrix heatmap

# Saves PNG files locally

import json
import csv
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# CONFIG
# -------------------------
PREDICTIONS_FILE = "evaluations/predictions2.csv"
METRICS_FILE = "evaluations/evaluation_metrics2.json"

image_name_confusion_matrix = "evaluations/confusion_matrix2.png"
image_name_metrics_barplot = "evaluations/metrics_barplot2.png"

# -------------------------
# LOAD PREDICTIONS CSV
# -------------------------
labels = []
predictions = []

with open(PREDICTIONS_FILE, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Convert strings to int
        labels.append(int(row["label"]))
        predictions.append(int(row["prediction"]))

print("Loaded predictions:", len(labels))

# -------------------------
# LOAD METRICS JSON
# -------------------------
with open(METRICS_FILE, "r") as f:
    metrics = json.load(f)

print("Loaded metrics")

# -------------------------
# VISUALIZATION 1:
# CONFUSION MATRIX HEATMAP
# -------------------------
cm = np.array(metrics["confusion_matrix"])

plt.figure(figsize=(6, 6))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix")
plt.colorbar()

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(image_name_confusion_matrix)
plt.close()

print(f"Saved {image_name_confusion_matrix}")

# -------------------------
# VISUALIZATION 2:
# PRECISION, RECALL, F1 BARPLOT
# -------------------------
plt.figure()
plt.bar(
    ["Precision", "Recall", "F1"],
    [
        metrics["precision_weighted"],
        metrics["recall_weighted"],
        metrics["f1_weighted"]
    ]
)
plt.ylim(0, 1)
plt.title("Weighted Metrics")
plt.tight_layout()
plt.savefig(image_name_metrics_barplot)
plt.close()

print(f"Saved {image_name_metrics_barplot}")

print("✅ All graphs generated!")
