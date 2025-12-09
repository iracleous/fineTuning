# Downloads data from the hugging face and saves them locally.

from datasets import load_dataset
from pathlib import Path

# --- Configuration ---
HF_DATASET_NAME = "imdb"
SAVE_DIR = "./local_data"

def download_and_save():
    """
    Downloads a dataset from Hugging Face and saves it locally
    in multiple formats (JSON, CSV, Parquet).
    """

    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    print(f"📥 Downloading dataset '{HF_DATASET_NAME}' ...")

    # Load full dataset
    dataset = load_dataset(HF_DATASET_NAME)

    print("✅ Dataset downloaded successfully!")

    # Save each split
    for split_name, split_data in dataset.items():
        print(f"💾 Saving split '{split_name}'...")

        # Save to JSON
        json_path = Path(SAVE_DIR) / f"{split_name}.json"
        split_data.to_json(json_path)
        
        # Save to CSV
        csv_path = Path(SAVE_DIR) / f"{split_name}.csv"
        split_data.to_csv(csv_path)

        # Save to Parquet
        parquet_path = Path(SAVE_DIR) / f"{split_name}.parquet"
        split_data.to_parquet(parquet_path)

        print(f"   → Saved JSON, CSV, Parquet for '{split_name}'")

    print("\n🎉 All dataset splits saved locally!")

if __name__ == "__main__":
    download_and_save()
