import asyncio

from llmscratch.config.config import Config
from llmscratch.exception import CustomException
from llmscratch.logger_manager import LoggerManager
from llmscratch.models.command_line_args import CommandLineArgs
from llmscratch.pipelines.preprocessing_pipeline import PreprocessingPipeline
from llmscratch.pipelines.tokenizer_training_pipeline import TokenizerTrainingPipeline

logging = LoggerManager.get_logger(__name__)


class Host:
    """
    Host class to manage the execution of the main application.

    This class handles initialization with command-line arguments and
    configuration, and runs the main asynchronous functionality.
    """

    def __init__(self, args: CommandLineArgs):
        self.args = args
        self.config = Config()

        # Load config from YAML if provided
        if args.config:
            self.config.config_path = args.config
            self.config.load_from_yaml(args.config)

        # Apply CLI overrides if any
        self.config.apply_cli_overrides(args)

    def run(self):
        return asyncio.run(self.run_async())

    async def run_async(self):
        try:
            logging.info("🚀 Starting host operations.")

            if self.args.command == "preprocess":
                logging.info("🧪 Executing LLM data preprocessing pipeline.")
                await self.run_preprocess()
            elif self.args.command == "train_tokenizer":
                logging.info("✍️ Executing tokenizer training pipeline.")
                await self.run_tokenizer_training()
            else:
                logging.error(f"❌ Unknown subcommand: {self.args.command}")
                raise ValueError(
                    "Please specify a valid subcommand: 'preprocess' or 'train_tokenizer'."
                )

        except CustomException as e:
            logging.error("🔥 A custom error occurred: %s", e)
            raise
        except Exception as e:
            logging.error("💥 An unexpected error occurred: %s", e)
            raise
        finally:
            logging.info("✅ Shutting down host gracefully.")

    async def run_preprocess(self):
        """
        Run the full preprocessing pipeline for LLM training.
        """
        try:
            logging.info("📦 Running preprocessing pipeline...")

            pipeline = PreprocessingPipeline()
            # For now, test with placeholder text or load actual data
            sample_text = "Hello world. This is a test sentence for tokenization."
            result_df = pipeline.run(sample_text)

            # Optional: save result to disk (e.g., as CSV or .npy)
            # if self.config.save_preprocessed_output:
            #     output_path = self.config.preprocessed_output_path
            #     result_df.to_csv(output_path, index=False)
            #     logging.info(f"📝 Saved output to: {output_path}")

            logging.info("✅ Preprocessing pipeline completed successfully.")

        except Exception as e:
            raise CustomException(e) from e

    async def run_tokenizer_training(self):
        """
        Run the tokenizer training pipeline.
        """
        try:
            logging.info("📚 Running tokenizer training pipeline...")

            pipeline = TokenizerTrainingPipeline()
            await pipeline.run()

            logging.info("✅ Tokenizer training completed successfully.")

        except Exception as e:
            raise CustomException(e) from e
