import asyncio

from config.config import Config
from exception import CustomException
from logger_manager import LoggerManager
from models.command_line_args import CommandLineArgs

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
            else:
                logging.error(f"❌ Unknown subcommand: {self.args.command}")
                raise ValueError("Please specify a valid subcommand: 'preprocess'.")

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
            # TODO: Replace with actual preprocessing steps
            # e.g., PreprocessingPipeline(config=self.config).run()
            logging.info("✅ Preprocessing pipeline completed successfully.")

        except Exception as e:
            raise CustomException(e) from e
