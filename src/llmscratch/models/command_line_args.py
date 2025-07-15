from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple


@dataclass
class CommandLineArgs:
    """
    Structured command-line arguments for the LLM from scratch.

    Supports:
    - Ingest pipeline
    """

    # === Core CLI ===
    command: str  # Subcommands:
    config: Optional[str] = None  # Optional path to YAML config file
    debug: bool = False  # Enable verbose logging
    _explicit_args: Set[str] = field(default_factory=set)
