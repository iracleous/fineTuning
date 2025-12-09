# Below is a clean, minimal script that does exactly :

# Loads data from a local folder (previously downloaded → JSON/CSV/Parquet).
# Loads the untrained pre-trained model from a local folder (downloaded earlier).
# Runs fine-tuning (training).
# It uses the same IMDB dataset columns:
# text
# label


import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score

# --- Paths ---
LOCAL_DATA_DIR = "./local_data"        # folder containing train.json / test.json
LOCAL_MODEL_DIR = "./local_model"      # folder containing untrained model
OUTPUT_DIR = "./results"

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
NUM_LABELS = 2
SEED = 42


# --------------------------
# 1. Load Local Dataset
# --------------------------
def load_local_dataset():
    print("📥 Loading dataset from local folder...")

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(Path(LOCAL_DATA_DIR) / "train.json"),
            "test": str(Path(LOCAL_DATA_DIR) / "test.json"),
        }
    )

    print("✅ Loaded local dataset:", dataset)
    return dataset


# --------------------------
# Metric Function
# --------------------------
def compute_metrics(p):
    preds, labels = p
    preds = preds.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average='weighted')
    }


# --------------------------
# 2. Load Local Model + Tokenizer
# --------------------------
def load_local_model():
    print("📥 Loading model from:", LOCAL_MODEL_DIR)

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        LOCAL_MODEL_DIR,
        num_labels=NUM_LABELS
    )

    print("✅ Model and tokenizer loaded from local storage!")
    return tokenizer, model


# --------------------------
# Tokenization Function
# --------------------------
def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples[TEXT_COLUMN],
        truncation=True,
        max_length=128
    )


# --------------------------
# 3. Run Training
# --------------------------
def run_training():
    # 1. Load dataset
    raw_datasets = load_local_dataset()

    # 2. Load local model
    tokenizer, model = load_local_model()

    # 3. Tokenize
    print("🔄 Tokenizing dataset...")
    tokenized = raw_datasets.map(
        lambda e: tokenize_function(e, tokenizer),
        batched=True
    )
    tokenized = tokenized.rename_column(LABEL_COLUMN, "labels")
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_ds = tokenized["train"]
    eval_ds = tokenized["test"]

    # 4. Training settings
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        seed=SEED,
        logging_dir="./logs",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )

    # 5. Train
    print("🚀 Starting training...")
    trainer.train()

    # 6. Evaluate
    print("📊 Final Evaluation:")
    print(trainer.evaluate())

    # 7. Save fine-tuned model
    output_path = Path("./fine_tuned_model")
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)

    print(f"🎉 Training complete! Fine-tuned model saved to {output_path}")


# -------------------
# MAIN ENTRY
# -------------------
if __name__ == "__main__":
    if torch.cuda.is_available():
        print("🚀 CUDA available:", torch.cuda.get_device_name(0))
    else:
        print("⚠ No GPU found. Training will run on CPU.")

    run_training()
