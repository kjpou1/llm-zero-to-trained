from typing import Optional

import pandas as pd

from llmscratch.config.config import Config
from llmscratch.logger_manager import LoggerManager

# from llmscratch.services.sampling_service import SamplingService
# from llmscratch.services.text_cleaning_service import TextCleaningService
# from llmscratch.services.tokenizer_service import TokenizerService

logging = LoggerManager.get_logger(__name__)


class PreprocessingPipeline:
    def __init__(self):
        self.config = Config()
        self._initialize_services()

    def _initialize_services(self):
        logging.info("[PreprocessingPipeline] Initializing services...")

        # self.cleaner = TextCleaningService(
        #     remove_non_ascii=self.config.remove_non_ascii
        # )

        # self.tokenizer = TokenizerService(
        #     vocab_path=self.config.vocab_path,
        #     merges_path=self.config.merges_path,
        #     special_tokens=self.config.special_tokens,
        # )

        # self.sampler = SamplingService(
        #     sequence_length=self.config.sequence_length,
        #     stride=self.config.stride,
        #     bos_token=self.config.special_tokens.get("bos"),
        #     eos_token=self.config.special_tokens.get("eos"),
        # )

    def _validate_input(self, text: str) -> bool:
        if not text or not isinstance(text, str):
            logging.error("[PreprocessingPipeline] Invalid input text")
            return False
        return True

    def run(self, raw_text: str) -> Optional[pd.DataFrame]:
        if not self._validate_input(raw_text):
            raise ValueError("Invalid raw text input")

        logging.info("[PreprocessingPipeline] Starting preprocessing pipeline...")

        # cleaned_text = self.cleaner.clean(raw_text)
        # token_ids = self.tokenizer.tokenize(cleaned_text)
        # samples = self.sampler.sample(token_ids)
        samples = pd.DataFrame()  # Placeholder for actual processing logic

        logging.info("[PreprocessingPipeline] Preprocessing complete")
        return samples

    def run_batch(self, texts: list[str]) -> list[Optional[pd.DataFrame]]:
        logging.info(
            f"[PreprocessingPipeline] Starting batch processing on {len(texts)} texts..."
        )
        results = []
        for i, text in enumerate(texts):
            logging.info(f"[PreprocessingPipeline] Processing item {i+1}/{len(texts)}")
            try:
                result = self.run(text)
                results.append(result)
            except Exception as e:
                logging.error(
                    f"[PreprocessingPipeline] Error processing item {i+1}: {e}"
                )
                results.append(None)
        return results
