# 🧠 Byte Pair Encoding (BPE) Implementation Guide

This document details our implementation of Byte Pair Encoding (BPE) as described in:

> Sennrich, Rico; Haddow, Barry; Birch, Alexandra (2015).  
> *Neural Machine Translation of Rare Words with Subword Units*  
> [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)

---

## 🎯 Objective

Implement a faithful version of BPE for tokenizing text into subword units, suitable for training a custom LLM from scratch. This implementation follows the original paper closely while introducing practical logging, vocabulary management, and modular service structure.

---

## 📦 File: `bpe_trainer.py`

### Key Components

- `fit(word_freqs: Dict[str, int])`  
  Main entry point to learn BPE merge rules over a vocabulary.

- `_init_vocab(word_freqs)`  
  Converts raw word frequency dict into character-level vocab with `</w>` markers.

- `_count_symbol_pairs(vocab)`  
  Counts how often each adjacent symbol pair occurs, weighted by word frequency.

- `max(symbol_pairs, key=symbol_pairs.get)`  
  Greedily selects the most frequent pair for merging.

- `_merge_vocab(pair, vocab)`  
  Uses regex to merge the selected pair in all words.

- `save_artifacts(output_path)`  
  Saves `vocab.txt` and `merges.txt` for downstream encoding.

---

## ⚙️ Overview of Algorithm (Paper Alignment)

| Step     | Description                                                          | Function                                  |
| -------- | -------------------------------------------------------------------- | ----------------------------------------- |
| ✅ Step 1 | Represent each word as a sequence of characters + `</w>`             | `_init_vocab()`                           |
| ✅ Step 2 | Count frequency of all adjacent symbol pairs (weighted by word freq) | `_count_symbol_pairs()`                   |
| ✅ Step 3 | **Select the most frequent symbol pair** to merge (greedy step)      | `max(symbol_pairs, key=symbol_pairs.get)` |
| ✅ Step 4 | Merge the selected pair across the entire vocab                      | `_merge_vocab()`                          |
| 🔁 Step 5 | Repeat Steps 2–4 for `N` merge operations                            | `fit()`                                   |
| 💾 Step 6 | Save the final vocabulary and merge rules                            | `save_artifacts()`                        |

---

## 📖 Paper Alignment Notes

### ✅ Step 1: Initial Vocab with `</w>`
> “We represent each word as a sequence of characters plus a special end-of-word symbol.”

✔ Done in `_init_vocab()`

---

### ✅ Step 2: Count Symbol Pairs
> “We iteratively count all symbol pairs…”

✔ Done in `_count_symbol_pairs()` using weighted frequency.

---

### ✅ Step 3: Select Most Frequent Pair (Greedy)
> “…and replace each occurrence of the most frequent pair…”

✔ Done using `max()` on the pair stats.

---

### ✅ Step 4: Merge the Pair
> “The symbol vocabulary is extended with the newly merged symbol.”

✔ Done via regex substitution in `_merge_vocab()`

---

## 🧪 Design Notes

- All vocabulary entries are represented as immutable `Tuple[str, ...]` symbols for dictionary safety.
- `</w>` is used to mark end-of-word boundaries, as in the original paper.
- Regex ensures precise merging with token boundaries:
  ```python
  pattern = re.compile(rf"(?<!\S){bigram}(?!\S)")
  ```

* Logs merge progress every 100 steps.
* Saves artifacts in simple text format:

  * `vocab.txt`: space-separated symbolized word and frequency.
  * `merges.txt`: list of merge operations (in order).

---

## 📊 Logging and Statistics

Call `trainer.log_statistics()` after training to log:

* ✅ Final vocab size
* ✅ Total merges performed
* ✅ Average symbols per word (frequency-weighted)

---

## 🛠️ Future Enhancements

* Add encode/decode routines for downstream tokenization
* Integrate fast inference-time tokenizer using saved merges
* Visualize merge hierarchy as a tree or DAG
* Add support for UnigramLM or WordPiece variations

---

## 📁 Artifacts

* `vocab.txt`
  Example line:

  ```
  the 2435
  he 812
  low 57
  ```

* `merges.txt`
  Example:

  ```
  t h
  th e</w>
  l ow
  ```
