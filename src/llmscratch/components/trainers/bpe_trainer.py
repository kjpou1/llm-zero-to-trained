import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

from llmscratch.components.trainers.base_trainer import BaseTrainer
from llmscratch.logger_manager import LoggerManager

logging = LoggerManager.get_logger(__name__)


class BPETrainer(BaseTrainer):
    """
    Learns BPE merge operations from a word frequency dictionary.
    """

    def __init__(self, num_merges: int = 10000):
        """
        Initializes the BPE trainer.

        Args:
            num_merges (int): The number of merge operations to perform. Controls subword vocab size.
        """
        self.num_merges = num_merges
        self.vocab: Dict[Tuple[str, ...], int] = {}
        self.merges: List[Tuple[str, str]] = []
        logging.info(f"📦 Initialized BPETrainer with num_merges={self.num_merges}")

    def fit(self, word_freqs: Dict[str, int]) -> List[Tuple[str, str]]:
        """
        Learns BPE merge rules from the given word frequency dictionary.

        Args:
            word_freqs (Dict[str, int]): Dictionary mapping words to frequencies.

        Returns:
            List[Tuple[str, str]]: List of merge operations in order.
        """
        logging.info("🚀 Starting BPE training...")
        self.vocab = self._init_vocab(word_freqs)
        self.merges = []  # Reset in case trainer is reused
        logging.info(f"🧱 Initialized vocab with {len(self.vocab)} entries.")

        for i in range(self.num_merges):
            symbol_pairs = self._count_symbol_pairs(self.vocab)

            if not symbol_pairs:
                logging.info(f"🛑 No more symbol pairs left after {i} merges.")
                break

            best_pair = max(symbol_pairs, key=symbol_pairs.get)
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

    def log_statistics(self) -> None:
        self._log_merge_statistics()

    def _init_vocab(self, word_freqs: Dict[str, int]) -> Dict[Tuple[str, ...], int]:
        """
        Convert word-level frequency dict into character-symbol vocab for BPE.

        Each word becomes a tuple of characters + a special end-of-word symbol `</w>`.

        From the BPE paper:
            “We represent each word as a sequence of characters plus a special end-of-word symbol, and merge the most frequent symbol pair.”

        Args:
            word_freqs (Dict[str, int]): Raw word frequencies.

        Returns:
            Dict[Tuple[str, ...], int]: Initialized vocab with character symbols.
        """
        vocab = {}
        for word, freq in word_freqs.items():
            symbols = tuple(list(word) + ["</w>"])
            vocab[symbols] = freq
        return vocab

    def save_artifacts(self, output_path: str) -> None:
        """
        Save BPE vocab and merges to the specified directory.

        Args:
            output_path (str): Directory to save `vocab.txt` and `merges.txt`.
        """
        os.makedirs(output_path, exist_ok=True)

        vocab_path = os.path.join(output_path, "vocab.txt")
        merges_path = os.path.join(output_path, "merges.txt")

        # Save vocab
        with open(vocab_path, "w", encoding="utf-8") as f:
            for word_tuple, freq in self.vocab.items():
                word = "".join(word_tuple).replace("</w>", "")  # optional cleanup
                f.write(f"{word} {freq}\n")

        # Save merges
        with open(merges_path, "w", encoding="utf-8") as f:
            for a, b in self.merges:
                f.write(f"{a} {b}\n")

        logging.info(f"📁 Saved vocab to: {vocab_path}")
        logging.info(f"📁 Saved merges to: {merges_path}")

    def _count_symbol_pairs(
        self, vocab: Dict[Tuple[str, ...], int]
    ) -> Dict[Tuple[str, str], int]:
        """
        Count frequency of all adjacent symbol pairs in the current BPE vocabulary.

        Each word is a tuple of symbols (initially characters + </w>).
        We count how many times each bigram (pair of symbols) appears,
        weighted by the frequency of the word in which it appears.

        Args:
            vocab (Dict[Tuple[str, ...], int]): Current BPE vocabulary.

        Returns:
            Dict[Tuple[str, str], int]: Symbol pair → frequency mapping.
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
        Merge the most frequent symbol pair throughout the vocabulary.

        Args:
            pair (Tuple[str, str]): The symbol pair to merge (e.g. ('t', 'h')).
            vocab (Dict[Tuple[str, ...], int]): Current vocabulary.

        Returns:
            Dict[Tuple[str, ...], int]: New vocabulary with merged symbols.
        """
        merged_vocab = {}
        bigram = re.escape(" ".join(pair))  # e.g. "t h" → regex safe
        pattern = re.compile(rf"(?<!\S){bigram}(?!\S)")  # match whole token pair only

        for word_tuple, freq in vocab.items():
            word_str = " ".join(word_tuple)
            # Replace the pair with merged symbol
            new_word_str = pattern.sub("".join(pair), word_str)
            new_word = tuple(new_word_str.split())
            merged_vocab[new_word] = freq

        return merged_vocab

    def _log_merge_statistics(self) -> None:
        """
        Logs statistics about the final vocabulary after BPE training.

        Includes:
        - Total unique words (vocab size)
        - Total symbols (weighted by frequency)
        - Average symbols per word
        - Number of merge operations performed
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
