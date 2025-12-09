# Below is a simple, clean example showing how to:

# Load your LOCAL model (untrained BERT).
# Load your LOCAL test dataset (e.g., test.json).
# Run inference on ONLY 3 rows
# Compute accuracy and F1 ONLY for those 3 predictions.


import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path

LOCAL_MODEL_DIR = "./local_model"       # untrained model folder
LOCAL_DATA_DIR = "./local_data"         # folder containing test.json
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
NUM_TEST_ROWS = 3


# ------------------------------------------------
# 1. Load 3 samples from local test data
# ------------------------------------------------
def load_test_samples():
    print("📥 Loading local test dataset...")

    dataset = load_dataset(
        "json",
        data_files={"test": str(Path(LOCAL_DATA_DIR) / "test.json")}
    )["test"]

    print(f"📘 Dataset loaded. Total rows: {len(dataset)}")
    print(f"🔎 Selecting first {NUM_TEST_ROWS} rows...\n")

    return dataset.select(range(NUM_TEST_ROWS))


# ------------------------------------------------
# 2. Load model + tokenizer from local folder
# ------------------------------------------------
def load_model():
    print("📥 Loading model from:", LOCAL_MODEL_DIR)

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        LOCAL_MODEL_DIR,
        num_labels=2
    )
    model.eval()

    print("✅ Model & tokenizer loaded successfully!\n")
    return tokenizer, model


# ------------------------------------------------
# 3. Run prediction on 3 samples
# ------------------------------------------------
def evaluate_three_rows():
    # Load model + tokenizer
    tokenizer, model = load_model()

    # Load 3 samples
    samples = load_test_samples()

    # Extract text and labels properly
    texts = [row[TEXT_COLUMN] for row in samples]
    true_labels = [row[LABEL_COLUMN] for row in samples]

    print("📘 Testing on the following 3 text samples:\n")
    for i, t in enumerate(texts):
        print(f"Row {i}: {t[:200]}...\n")

    # Tokenize inputs
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).tolist()

    # Metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')

    # Print results
    print("\n🔎 **Predictions:**")
    for i in range(NUM_TEST_ROWS):
        print(f"Row {i}: true={true_labels[i]}  |  pred={predictions[i]}")
    
    print("\n📊 **Evaluation on 3 rows**")
    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("\nDone.")


# ------------------------------------------------
# MAIN
# ------------------------------------------------
if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"🚀 CUDA GPU detected: {torch.cuda.get_device_name(0)}\n")
    else:
        print("⚠ No GPU found, running on CPU.\n")

    evaluate_three_rows()
