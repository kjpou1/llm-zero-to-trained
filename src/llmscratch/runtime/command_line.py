import sys

from llmscratch.models.command_line_args import CommandLineArgs

from .logging_argument_parser import LoggingArgumentParser


class CommandLine:
    @staticmethod
    def parse_arguments() -> CommandLineArgs:
        """
        Parse command-line arguments and return a CommandLineArgs object.

        Supports subcommands like 'ingest', 'train', 'inference', and 'render-images'.
        """
        parser = LoggingArgumentParser(
            description="Convolution CV CLI for chart pattern ingestion, model training, image inference, and image rendering workflows."
        )

        # Create subparsers for subcommands
        subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

        # === INGEST Subcommand ===
        preprocess_parser = subparsers.add_parser(
            "preprocess", help="Run the preprocess pipeline"
        )
        preprocess_parser.add_argument(
            "--config", type=str, help="Path to YAML config file"
        )
        preprocess_parser.add_argument(
            "--debug", action="store_true", help="Enable debug logging"
        )

        # Parse the arguments
        args = parser.parse_args()

        # Check if a subcommand was provided
        if args.command is None:
            parser.print_help()
            exit(1)

        # Determine which subparser was used
        command = args.command

        # Get subparser object based on command
        subparser = {
            "preprocess": preprocess_parser,
        }.get(command)

        # Track which args were explicitly passed on CLI
        args._explicit_args = set()
        if subparser:
            for action in subparser._actions:
                for opt in action.option_strings:
                    if opt in sys.argv:
                        args._explicit_args.add(action.dest)

        # If config is NOT specified, validate that required CLI arguments are present
        if args.command == "preprocess":
            CommandLine._validate_preprocess_args(args, parser)

        # Return a CommandLineArgs object with parsed values
        return CommandLineArgs(
            command=args.command,
            _explicit_args=getattr(args, "_explicit_args", set()),
            config=getattr(args, "config", None),
            debug=getattr(args, "debug", False),
        )

    @staticmethod
    def _validate_preprocess_args(args, parser):
        """
        Validate required preprocess arguments if no config is provided.
        """
        pass
