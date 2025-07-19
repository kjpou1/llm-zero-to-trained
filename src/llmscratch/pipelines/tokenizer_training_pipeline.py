import os

from llmscratch.components.trainers.bpe_trainer import BPETrainer
from llmscratch.config.config import Config
from llmscratch.logger_manager import LoggerManager
from llmscratch.services.text_preprocessing_service import TextPreprocessingService

logging = LoggerManager.get_logger(__name__)


class TokenizerTrainingPipeline:
    """
    Pipeline for training a BPE tokenizer from raw text.
    """

    def __init__(self):
        self.config = Config()
        self.text_service = TextPreprocessingService(self.config.DATASETS_RAW_DIR)
        self.trainer = BPETrainer(num_merges=self.config.tokenizer_merges)

    async def run(self):
        logging.info("🛠️ Starting tokenizer training pipeline...")

        # Step 1: Build word frequency vocabulary
        word_freq = self.text_service.build_word_freq(
            mode="line", lowercase=self.config.lowercase
        )

        # Step 2: Train BPE and collect merges
        merges = self.trainer.fit(word_freq)

        # Step 3: Save artifacts
        output_dir = self.config.tokenizer_output_dir
        if not output_dir:
            logging.warning(
                "⚠️  tokenizer_output_dir is not set. Skipping save_artifacts()."
            )
        else:
            try:
                os.makedirs(output_dir, exist_ok=True)
                self.trainer.save_artifacts(output_path=output_dir)
            except Exception as e:
                logging.warning(f"⚠️  Failed to save artifacts to {output_dir}: {e}")

        logging.info("✅ Tokenizer training pipeline complete.")
