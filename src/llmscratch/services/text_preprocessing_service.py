import os
from collections import Counter
from typing import Dict, List

import spacy

from llmscratch.components.io.text_loader import TextLoader
from llmscratch.logger_manager import LoggerManager

logging = LoggerManager.get_logger(__name__)


class TextPreprocessingService:
    """
    Service to preprocess text data and generate a word frequency dictionary.
    """

    def __init__(self, input_dir: str):
        """
        Args:
            input_dir (str): Directory containing raw .txt files.
        """
        self.input_dir = input_dir
        self.text_loader = TextLoader()

        logging.info("⚙️ Initializing spaCy language model...")
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger"])
        logging.info("✅ spaCy model loaded.")

    def split_line_into_words(self, line: str, lowercase: bool = False) -> List[str]:
        """
        Splits a line of text into clean words using spaCy.

        Only alphabetic tokens are retained. Lowercasing is optional.

        Args:
            line (str): Raw input line.
            lowercase (bool): If True, convert words to lowercase. Default is False.

        Returns:
            List[str]: A list of cleaned words from the input line.
        """
        doc = self.nlp(line)
        if lowercase:
            words = [token.text.lower() for token in doc if token.is_alpha]
        else:
            words = [token.text for token in doc if token.is_alpha]

        logging.debug(f"🔤 Extracted words: {words}")
        return words

    def build_word_freq(self, mode="line", lowercase: bool = False) -> Dict[str, int]:
        """
        Builds a word frequency dictionary from all .txt files in input_dir.

        Returns:
            Dict[str, int]: Mapping from word → frequency.
        """
        word_counter = Counter()

        logging.info(f"📁 Scanning directory: {self.input_dir}")
        for filename in os.listdir(self.input_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.input_dir, filename)
                logging.info(f"📄 Processing file: {filename}")
                for line in self.text_loader.load_text(file_path, mode=mode):
                    words = self.split_line_into_words(line, lowercase=lowercase)
                    word_counter.update(words)

        logging.info(
            f"✅ Word frequency dictionary built with {len(word_counter)} unique words."
        )
        return dict(word_counter)
