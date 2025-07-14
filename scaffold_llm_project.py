import os

BASE_DIR = "."

structure = {
    "notebooks": [
        "01_data_preparation.ipynb",
        "02_tokenizer.ipynb",
        "03_transformer.ipynb",
        "04_training_loop.ipynb",
    ],
    "tokenizer": [
        "__init__.py",
        "bpe_tokenizer.py",
        "wordpiece_tokenizer.py",
        "unigram_tokenizer.py",
    ],
    "model": [
        "__init__.py",
        "attention.py",
        "transformer_block.py",
        "model_config.py",
    ],
    "data": [
        "__init__.py",
        "dataset_loader.py",
        "batch_generator.py",
    ],
    "train": [
        "__init__.py",
        "trainer.py",
        "optimizer.py",
        "losses.py",
    ],
    "experiments": [],
    "configs": [
        "default_config.yaml",
    ],
}

# Create root-level files if not present
readme_path = os.path.join(BASE_DIR, "README.md")
if not os.path.exists(readme_path):
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("# LLM: Zero to Trained\n\nComing soon...\n")

progress_path = os.path.join(BASE_DIR, "PROGRESS.md")
if not os.path.exists(progress_path):
    with open(progress_path, "w", encoding="utf-8") as f:
        f.write("# Progress Log\n\n- [ ] Project initialized\n")

# Create directory structure and files
for folder, files in structure.items():
    dir_path = os.path.join(BASE_DIR, folder)
    os.makedirs(dir_path, exist_ok=True)
    for file in files:
        file_path = os.path.join(dir_path, file)
        if file.endswith(".ipynb"):
            # Minimal empty notebook
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(
                    '{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}'
                )
        elif file.endswith(".py"):
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"# {file}\n")
        elif file.endswith(".yaml"):
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("# default_config.yaml\n")

print("✅ Project scaffold created in current repo directory.")
