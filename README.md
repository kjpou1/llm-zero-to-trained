# LLM: Zero to Trained

This project is a fully manual, from-scratch implementation of a Large Language Model (LLM), guided by the book [**Build a Large Language Model from Scratch**](https://github.com/rasbt/LLMs-from-scratch) by Sebastian Raschka.

Unlike simply running code from the reference repo, the goal here is to **understand and re-implement every major component** — tokenizer, model, data loader, training loop — by hand.

---

- [LLM: Zero to Trained](#llm-zero-to-trained)
  - [🎯 Objective](#-objective)
  - [📁 Project Structure](#-project-structure)
  - [🧠 Learning Sources](#-learning-sources)
  - [🗒️ Progress Log](#️-progress-log)
  - [🔧 Getting Started](#-getting-started)
    - [🧰 Environment Setup (with `uv`)](#-environment-setup-with-uv)
    - [🚀 Start Building](#-start-building)
  - [📚 References \& Inspirations](#-references--inspirations)
  - [📜 License](#-license)

---

## 🎯 Objective

To develop a working LLM from first principles by:

- Writing each component from scratch in Python
- Following the structure and logic from the book
- Validating ideas through notebooks and experiments
- Building modular, reusable code

---

## 📁 Project Structure

```

notebooks/      ← Book-aligned exploration notebooks
tokenizer/      ← BPE, WordPiece, and Unigram tokenizer implementations
model/          ← Transformer architecture components
data/           ← Dataset loading, preprocessing, batching utilities
train/          ← Training loop, loss, optimizer, evaluation
experiments/    ← Saved runs, checkpoints, metrics from training runs
configs/        ← YAML configs for models and training (optional)
README.md       ← This file
PROGRESS.md     ← Running log of completed work and learnings

````

---

## 🧠 Learning Sources

- [LLMs-from-scratch GitHub Repo](https://github.com/rasbt/LLMs-from-scratch)
- Book: *Build a Large Language Model from Scratch* by Sebastian Raschka

---

## 🗒️ Progress Log

Check [`PROGRESS.md`](PROGRESS.md) for detailed notes and milestones.

---

## 🔧 Getting Started

Clone this repo, then explore and build each component:

```bash
git clone https://github.com/kjpou1/llm-zero-to-trained.git
cd llm-zero-to-trained
```

---

### 🧰 Environment Setup (with `uv`)

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management and reproducible environments.

1. **Install `uv`**
   👉 Follow the instructions here:
   [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

2. **Create and activate your virtual environment**:

```bash
uv venv
source .venv/bin/activate  # macOS/Linux
# OR
.venv\Scripts\activate     # Windows
```

3. **Sync dependencies**:

```bash
uv sync
```

This installs all required packages and generates a `uv.lock` file for reproducibility.

---

### 🚀 Start Building

Begin with:

* `notebooks/01_data_preparation.ipynb` – for exploring the dataset and tokenizer
* `tokenizer/` – to start implementing your own tokenizer logic
* `model/` – when you're ready to dive into Transformers


---

## 📚 References & Inspirations

This project is built entirely from scratch, with a strong emphasis on personal understanding through implementation.

The main learning resource is:

* [*Build a Large Language Model from Scratch* by Sebastian Raschka](https://github.com/rasbt/LLMs-from-scratch) — the primary guide followed throughout this project.

Additional projects that may inform future architectural decisions:

* [Karpathy’s minGPT and nanoGPT](https://github.com/karpathy/nanoGPT) — minimalist and educational GPT implementations
* [Hugging Face Transformers](https://github.com/huggingface/transformers) — a modular, production-ready reference for LLM design

These references serve as conceptual guides, but all implementation in this repository is original and written by hand.

---

## 📜 License

This project is MIT licensed.

