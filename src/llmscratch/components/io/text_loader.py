from typing import Iterator

from llmscratch.logger_manager import LoggerManager

logging = LoggerManager.get_logger(__name__)


class TextLoader:
    """
    A utility class for loading text files line-by-line or in chunks.

    This class abstracts the loading strategy for small and large files.
    Useful when building tokenizers that must operate on arbitrary corpus sizes.
    """

    def load_text(self, file_path: str, mode: str = "line") -> Iterator[str]:
        """
        Loads text from a file using the specified strategy.

        Args:
            file_path (str): Full path to the .txt file to read.
            mode (str): 'line' (default) to yield individual lines.
                        'chunk' to yield fixed-size byte chunks (for large files).

        Yields:
            str: A line (if mode="line") or a block of raw text (if mode="chunk").
        """
        logging.info(f"📂 Loading file: {file_path} (mode: {mode})")

        if mode == "line":
            yield from self._read_lines(file_path)
        elif mode == "chunk":
            logging.warning(
                f"⚠️ Chunk mode may split words across chunk boundaries. Not recommended for BPE or word-level NLP preprocessing."
            )
            yield from self._read_chunks(file_path)
        else:
            logging.error(f"❌ Unsupported mode: {mode}")
            raise ValueError(f"Unsupported mode: {mode}. Use 'line' or 'chunk'.")

    def _read_lines(self, file_path: str) -> Iterator[str]:
        """
        Internal method to yield lines of text stripped of newline characters.
        """
        logging.debug(f"🔍 Reading file line-by-line: {file_path}")
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                yield line.strip()

    def _read_chunks(self, file_path: str, chunk_size: int = 8192) -> Iterator[str]:
        """
        Internal method to yield fixed-size chunks of raw text from file.

        Args:
            chunk_size (int): Number of bytes to read per chunk. Default is 8KB.
        """
        logging.debug(
            f"🔍 Reading file in chunks: {file_path} (chunk size: {chunk_size} bytes)"
        )
        with open(file_path, encoding="utf-8") as f:
            while chunk := f.read(chunk_size):
                yield chunk
