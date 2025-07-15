# **LLM: Zero to Trained**

> A fully manual, from-scratch implementation of a Large Language Model (LLM), guided by the book [**Build a Large Language Model from Scratch**](https://github.com/rasbt/LLMs-from-scratch) by Sebastian Raschka.

Unlike simply running code from the [reference repo](https://github.com/rasbt/LLMs-from-scratch.git), this project is about **understanding and re-implementing** every major component — tokenizer, model, data loader, training loop — by hand.

---
- [**LLM: Zero to Trained**](#llm-zero-to-trained)
  - [🎯 Objective](#-objective)
  - [📁 Project Structure](#-project-structure)
  - [🧠 Learning Sources](#-learning-sources)
  - [🗒️ Progress Log](#️-progress-log)
  - [🔧 Getting Started](#-getting-started)
  - [🧰 Environment Setup (with uv)](#-environment-setup-with-uv)
  - [🚀 CLI: Start Building](#-cli-start-building)
  - [📚 References \& Inspirations](#-references--inspirations)
  - [📜 License](#-license)

---

## 🎯 Objective

To develop a working LLM from first principles by:

* Writing each component from scratch in Python
* Following the structure and logic from Raschka’s book
* Validating ideas through notebooks and experiments
* Building modular, reusable, CLI-driven code

---

## 📁 Project Structure

```
llm-zero-to-trained/
├── src/llmscratch/        ← Modular CLI-driven Python package
│   ├── config/            ← Config loader with .env, CLI, YAML support
│   ├── models/            ← Core dataclasses and SingletonMeta
│   ├── runtime/           ← CLI argument parsing and dispatch
│   ├── launch_host.py     ← Entry point for all commands (e.g., preprocess)
│   └── host.py            ← Command execution coordinator
├── notebooks/             ← Book-aligned exploration notebooks
├── configs/               ← YAML configs for datasets, vocab, etc.
├── datasets/              ← Raw and processed tokenized data
├── pyproject.toml         ← Project metadata and CLI definition
├── README.md              ← This file
└── PROGRESS.md            ← Running log of milestones
```

---

## 🧠 Learning Sources

* 📘 *Build a Large Language Model from Scratch* – Sebastian Raschka (2024)
* 💻 [LLMs-from-scratch GitHub Repo](https://github.com/rasbt/LLMs-from-scratch)
* 🧠 Karpathy’s [minGPT](https://github.com/karpathy/minGPT) and [nanoGPT](https://github.com/karpathy/nanoGPT) (inspirational, but not reused)

---

## 🗒️ Progress Log

See [PROGRESS.md](./PROGRESS.md) for completed milestones, model checkpoints, and active development notes.

---

## 🔧 Getting Started

```bash
git clone https://github.com/kjpou1/llm-zero-to-trained.git
cd llm-zero-to-trained
```

---

## 🧰 Environment Setup (with [uv](https://docs.astral.sh/uv/getting-started/installation/))

```bash
uv venv
source .venv/bin/activate        # macOS/Linux
# OR
.venv\Scripts\activate           # Windows

uv pip install --editable .
uv sync
```

> 🧪 This enables the `llmscratch` CLI from anywhere and installs all dependencies with reproducible locking via `uv`.

---

## 🚀 CLI: Start Building

Use the CLI to run modular LLM pipelines:

```bash
llmscratch preprocess --config configs/data_config.yaml
```

More commands like `train`, `sample`, and `evaluate` will follow as the project evolves.

---

## 📚 References & Inspirations

While the architecture is influenced by great projects, all code is original and written from scratch:

* [Raschka’s LLM Book](https://leanpub.com/llms-from-scratch)
* [minGPT](https://github.com/karpathy/minGPT)
* [nanoGPT](https://github.com/karpathy/nanoGPT)
* [Hugging Face Transformers](https://github.com/huggingface/transformers)

---

## 📜 License

This project is MIT licensed.
