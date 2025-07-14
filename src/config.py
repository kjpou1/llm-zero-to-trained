import os

# Project Root: assumes running from project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Dataset paths
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")
RAW_DIR = os.path.join(DATASETS_DIR, "raw")
PROCESSED_DIR = os.path.join(DATASETS_DIR, "processed")
VOCAB_DIR = os.path.join(DATASETS_DIR, "vocab")

# Auto-create folders if needed
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(VOCAB_DIR, exist_ok=True)


def print_config_info():
    print("=" * 50)
    print("📂 Configured Dataset Directories")
    print("-" * 50)
    print(f"{'Raw data dir:':25} {RAW_DIR}")
    print(f"{'Processed data dir:':25} {PROCESSED_DIR}")
    print(f"{'Vocab dir:':25} {VOCAB_DIR}")
    print("=" * 50)


# Optional: for direct CLI use
if __name__ == "__main__":
    print_config_info()
