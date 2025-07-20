import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

from llmscratch.components.trainers.base_trainer import BaseTrainer
from llmscratch.logger_manager import LoggerManager

logging = LoggerManager.get_logger(__name__)


class BPETrainer(BaseTrainer):
    """
    Byte Pair Encoding (BPE) Trainer — based on:
    "Neural Machine Translation of Rare Words with Subword Units" (Sennrich et al., 2016)

    This trainer learns merge rules from word frequencies by iteratively merging the most frequent pair of symbols
    in a vocabulary of character sequences augmented with a special end-of-word marker `</w>`.
    """

    def __init__(self, num_merges: int = 10000):
        """
        Initializes the BPE trainer.

        Args:
            num_merges (int): Number of BPE merge operations to perform.
                              More merges = larger subword vocabulary.
        """
        self.num_merges = num_merges
        self.vocab: Dict[Tuple[str, ...], int] = {}
        self.merges: List[Tuple[str, str]] = []
        logging.info(f"📦 Initialized BPETrainer with num_merges={self.num_merges}")

    def fit(self, word_freqs: Dict[str, int]) -> List[Tuple[str, str]]:
        """
        Main BPE training loop:
        - Convert word frequencies into character + `</w>` vocab.
        - Iteratively find and merge the most frequent adjacent symbol pair.

        From the paper:
            "We represent each word as a sequence of characters plus a special end-of-word symbol,
             and merge the most frequent symbol pair (‘A’, ‘B’) in the corpus, replacing each
             occurrence of the pair with a new symbol ‘AB’."

        Args:
            word_freqs (Dict[str, int]): Dictionary mapping words to frequencies.

        Returns:
            List[Tuple[str, str]]: Ordered list of BPE merge operations.
        """
        logging.info("🚀 Starting BPE training...")
        self.vocab = self._init_vocab(word_freqs)
        self.merges = []
        logging.info(f"🧱 Initialized vocab with {len(self.vocab)} entries.")

        for i in range(self.num_merges):
            symbol_pairs = self._count_symbol_pairs(self.vocab)

            if not symbol_pairs:
                logging.info(f"🛑 No more symbol pairs left after {i} merges.")
                break

            # Select most frequent pair
            best_pair = max(symbol_pairs, key=symbol_pairs.get)

            # Replace best_pair everywhere in the vocab
            self.vocab = self._merge_vocab(best_pair, self.vocab)
            self.merges.append(best_pair)

            if i % 100 == 0 or i == self.num_merges - 1:
                logging.info(
                    f"🔁 Merge {i+1}/{self.num_merges}: {best_pair} → {''.join(best_pair)}"
                )

        logging.info(
            f"✅ Training complete. Learned {len(self.merges)} merge operations."
        )
        return self.merges

    def _init_vocab(self, word_freqs: Dict[str, int]) -> Dict[Tuple[str, ...], int]:
        """
        Step 1: Initialize vocabulary.
        Each word is converted into a tuple of characters plus a special end-of-word marker `</w>`.

        Example:
            Input word = "lower"
            Output tuple = ('l', 'o', 'w', 'e', 'r', '</w>')

        Args:
            word_freqs (Dict[str, int]): Input word frequency dictionary.

        Returns:
            Dict[Tuple[str, ...], int]: Initial symbolized vocabulary.
        """
        vocab = {}
        for word, freq in word_freqs.items():
            symbols = tuple(list(word) + ["</w>"])
            vocab[symbols] = freq
        return vocab

    def _count_symbol_pairs(
        self, vocab: Dict[Tuple[str, ...], int]
    ) -> Dict[Tuple[str, str], int]:
        """
        Step 2: Count symbol pairs.
        For each word in the vocab, count the frequency of every adjacent symbol pair,
        weighted by the word's frequency.

        This is the preparatory step described in the BPE paper:
            "We iteratively count all symbol pairs..."

        The replacement or merge of the most frequent pair is handled in a separate step called _merge_vocab()
        pair_counts = defaultdict(int)

        Args:
            vocab (Dict[Tuple[str, ...], int]): Current symbolized vocabulary.

        Returns:
            Dict[Tuple[str, str], int]: Frequency counts of all adjacent pairs.
        """
        pair_counts = defaultdict(int)

        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] += freq
        return dict(pair_counts)

    def _merge_vocab(
        self, pair: Tuple[str, str], vocab: Dict[Tuple[str, ...], int]
    ) -> Dict[Tuple[str, ...], int]:
        """
        Step 3: Merge the best pair throughout the vocab.

        Implements the merge operation described in the BPE paper:
            “We iteratively count all symbol pairs and replace each occurrence of
             the most frequent pair (‘A’, ‘B’) with a new symbol ‘AB’.”

        This function finds all occurrences of the most frequent symbol pair in the vocabulary
        and replaces them with their merged form (e.g., ('t', 'h') → 'th').

        Args:
            pair (Tuple[str, str]): Most frequent symbol pair to merge.
            vocab (Dict[Tuple[str, ...], int]): Current vocabulary.

        Returns:
            Dict[Tuple[str, ...], int]: Updated vocabulary with merged pair.
        """
        merged_vocab = {}
        bigram = re.escape(" ".join(pair))  # Regex-safe
        pattern = re.compile(rf"(?<!\S){bigram}(?!\S)")

        for word_tuple, freq in vocab.items():
            word_str = " ".join(word_tuple)
            new_word_str = pattern.sub("".join(pair), word_str)
            new_word = tuple(new_word_str.split())
            merged_vocab[new_word] = freq

        return merged_vocab

    def save_artifacts(self, output_path: str) -> None:
        """
        Save learned BPE vocabulary and merge rules to disk.

        Args:
            output_path (str): Output directory.
        """
        os.makedirs(output_path, exist_ok=True)
        vocab_path = os.path.join(output_path, "vocab.txt")
        merges_path = os.path.join(output_path, "merges.txt")

        with open(vocab_path, "w", encoding="utf-8") as f:
            for word_tuple, freq in self.vocab.items():
                word = "".join(word_tuple).replace("</w>", "")
                f.write(f"{word} {freq}\n")

        with open(merges_path, "w", encoding="utf-8") as f:
            for a, b in self.merges:
                f.write(f"{a} {b}\n")

        logging.info(f"📁 Saved vocab to: {vocab_path}")
        logging.info(f"📁 Saved merges to: {merges_path}")

    def log_statistics(self) -> None:
        """Logs final vocabulary size and compression info."""
        self._log_merge_statistics()

    def _log_merge_statistics(self) -> None:
        """
        Logs vocabulary compression metrics after BPE training:
        - Number of entries in final vocab
        - Merge operations performed
        - Avg symbols per word (weighted by frequency)
        """
        if not self.vocab:
            logging.warning("⚠️ No vocabulary found. Run `fit()` first.")
            return

        total_entries = len(self.vocab)
        total_freq = sum(self.vocab.values())
        total_symbols = sum(len(word) * freq for word, freq in self.vocab.items())
        avg_symbols_per_word = total_symbols / total_freq if total_freq else 0

        logging.info("📊 BPE Merge Statistics:")
        logging.info(f"🔢 Unique symbolized words in final vocab: {total_entries}")
        logging.info(f"🔁 Merge operations learned: {len(self.merges)}")
        logging.info(f"🧱 Avg symbols per word (weighted): {avg_symbols_per_word:.2f}")
