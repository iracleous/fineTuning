import torch
from pathlib import Path
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score

# --- Configuration Section ---
MODEL_NAME = "bert-base-uncased" 
NUM_LABELS = 2
OUTPUT_DIR = "./results"
SEED = 42

# --- UPDATED: Hugging Face Dataset Configuration ---
HF_DATASET_NAME = "imdb" 
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

# --- Helper Functions ---

def compute_metrics(p):
    """
    Computes accuracy and F1 score for classification tasks.
    """
    predictions, labels = p
    predictions = torch.argmax(torch.tensor(predictions), dim=1).numpy()
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {"accuracy": accuracy, "f1": f1}


# --- Main Fine-Tuning Script ---

def run_fine_tuning():
    """
    Orchestrates the data preparation, model loading, and training process
    by loading data directly from the Hugging Face Hub.
    """
    # 1. Directory Setup 
    print("--- 1. Model & Output Setup ---")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True) 

    # 2. Load Dataset DIRECTLY from the Hugging Face Hub
    print(f"\n--- 2. Load and Tokenize Data from {HF_DATASET_NAME} ---")
    try:
        # Load the IMDB dataset and automatically create the train/validation splits
        raw_datasets: DatasetDict = load_dataset(
            HF_DATASET_NAME, 
            split={
                "train": "train[:90%]",      # 90% of IMDB train split for training
                "validation": "train[90%:]"  # 10% of IMDB train split for validation
            }
        )
        print(f"Dataset loaded with splits: {list(raw_datasets.keys())}")
        
    except Exception as e:
        print(f"❌ Failed to load dataset '{HF_DATASET_NAME}' from the Hub: {e}")
        return

    # 3. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize_function(examples):
        return tokenizer(examples[TEXT_COLUMN], truncation=True, max_length=128) 

    # 4. Apply Tokenization
    tokenized_datasets = raw_datasets.map(
        tokenize_function, 
        batched=True,
        # Keep only the tokenized inputs and the label column
        remove_columns=[col for col in raw_datasets["train"].column_names if col not in [LABEL_COLUMN]]
    )
    tokenized_datasets = tokenized_datasets.rename_column(LABEL_COLUMN, "labels")

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]
    
    print("Tokenization complete.")

    # 5. Model Loading
    print("\n--- 3. Load Model and Define Arguments ---")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=NUM_LABELS
    )

    # 6. Define Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir='./logs',
        logging_steps=50,
        seed=SEED
    )

    # 7. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 8. Start Training
    print("\n--- 4. Start Fine-Tuning (May take some time) ---")
    trainer.train()

    # 9. Final Evaluation and Save
    print("\n--- 5. Evaluation & Save ---")
    results = trainer.evaluate()
    print(f"Final Evaluation Results: {results}")

    final_model_path = Path("./fine_tuned_model")
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))
    print(f"\n✅ Fine-tuned model saved successfully to: {final_model_path.resolve()}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"CUDA GPU found: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA GPU not found. Training will run on CPU, which may be very slow.")
        
    run_fine_tuning()