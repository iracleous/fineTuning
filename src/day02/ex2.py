# dowonload the open source untrained model from https://huggingface.co/EleutherAI/gpt-neo-1.3B

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path

MODEL_NAME = "bert-base-uncased"
SAVE_DIR = "./local_model"

def download_model():
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

    print(f"📥 Downloading model '{MODEL_NAME}'...")

    # Download model weights
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.save_pretrained(SAVE_DIR)

    # Download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(SAVE_DIR)

    print(f"✅ Model and tokenizer saved to: {SAVE_DIR}")

if __name__ == "__main__":
    download_model()
