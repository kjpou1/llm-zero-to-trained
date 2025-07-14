Here is the updated version of `docs/structure.md` including the **cleanly 
# 📦 Project Structure Overview

This document explains the reasoning and sources behind the structure of the `llm-zero-to-trained` project.

---

## 🧠 Design Goal

Build a complete Large Language Model (LLM) from first principles — **not by running predefined scripts**, but by **writing every component by hand** using clean, modular Python code.

---

## 📚 Structure Origins

### 1. 🧱 Book: *Build a Large Language Model from Scratch* by Sebastian Raschka
- The book guides learners step-by-step through tokenizer design, transformer implementation, and training loops.
- Chapters are supported by focused **Jupyter notebooks**, which inspired the `notebooks/` directory.

### 2. 🛠️ General ML Engineering Practices
- Common structure patterns from modern LLM implementations encourage:
  - Clear separation of tokenizer/model/train/data logic
  - Experiment logging and configuration via versioned files
  - Reusability and scalability of components
- These ideas guided the modular organization of this project but **do not reflect direct code reuse**.

---

## 📁 Directory Breakdown

```

llm-zero-to-trained/
├── notebooks/          ← Book-aligned exploration notebooks
├── tokenizer/          ← BPE, WordPiece, Unigram tokenizers
├── model/              ← Transformer blocks, attention, embeddings
├── data/               ← Dataset loading, preprocessing, batching
├── train/              ← Training loop, loss, optimization
├── experiments/        ← Saved logs, checkpoints, run configs
├── configs/            ← YAML configs for models and training
├── README.md           ← Project overview and getting started
└── PROGRESS.md         ← Running log of implementation + insights

```

---

## 🔁 Why This Structure?

| Folder         | Purpose                                                                         |
| -------------- | ------------------------------------------------------------------------------- |
| `notebooks/`   | Aligns with the book’s learning path, enables testing ideas before modularizing |
| `tokenizer/`   | Each tokenizer (BPE, WordPiece, Unigram) is a standalone, testable component    |
| `model/`       | Transformer internals modularized for clarity and reuse                         |
| `data/`        | Centralizes all input pipelines, token ID conversion, batching                  |
| `train/`       | Isolates training-specific logic (optimizer, evaluation, checkpointing)         |
| `experiments/` | Encourages logging and version control of training runs                         |
| `configs/`     | Supports config-driven experiments for flexibility and reproducibility          |
| `PROGRESS.md`  | Tracks milestones, bugs, fixes, and learning notes                              |
| `README.md`    | Serves as the public-facing orientation and quickstart guide                    |

---

## 📚 References & Inspirations

The **primary educational guide** for this project is:

- [_Build a Large Language Model from Scratch_ by Sebastian Raschka](https://github.com/rasbt/LLMs-from-scratch) — the project closely follows the concepts and implementation flow presented in the book and companion repository.

Additional projects that **may inform future architectural decisions** include:

- [Karpathy’s minGPT and nanoGPT](https://github.com/karpathy/nanoGPT) — known for their minimalist, educational approach to Transformer implementation
- [Hugging Face Transformers](https://github.com/huggingface/transformers) — valuable for understanding modular, scalable, production-grade design patterns

All components are implemented from scratch, with a focus on hands-on understanding and independent development.
