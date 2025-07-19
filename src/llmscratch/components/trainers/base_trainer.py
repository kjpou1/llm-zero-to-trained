from abc import ABC, abstractmethod
from typing import Any


class BaseTrainer(ABC):
    """
    Abstract base class for tokenization trainers.
    """

    @abstractmethod
    def fit(self, data: Any) -> Any:
        """
        Learns from input data and returns learned artifacts
        (e.g., merge rules for BPE, vocab for WordPiece).

        Args:
            data (Any): Training input, such as word frequency dictionary.

        Returns:
            Any: Learned model (e.g., list of merges, vocabulary, etc.)
        """
        pass

    @abstractmethod
    def log_statistics(self) -> None:
        """
        Logs model-specific training statistics.
        Each subclass should override this.
        """
        pass

    @abstractmethod
    def save_artifacts(self, output_path: str) -> None:
        """
        Save all tokenizer training artifacts to the specified path.

        Args:
            output_path (str): Base directory where artifacts will be saved.
        """
        pass
