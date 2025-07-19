import os
from typing import Optional

import yaml
from dotenv import load_dotenv

from llmscratch.models import SingletonMeta
from llmscratch.utils.arg_utils import was_explicit as _was_explicit
from llmscratch.utils.path_utils import ensure_all_dirs_exist, get_project_root


class Config(metaclass=SingletonMeta):
    _is_initialized = False

    def __init__(self):
        if Config._is_initialized:
            return

        load_dotenv()

        self.PROJECT_ROOT = get_project_root()

        # === Dataset directory (read-only access) ===
        self.DATASETS_DIR = os.path.join(self.PROJECT_ROOT, "datasets")
        self.DATASETS_RAW_DIR = os.path.join(self.DATASETS_DIR, "raw")
        self.DATASETS_PROCESSED_DIR = os.path.join(self.DATASETS_DIR, "processed")

        # === Artifacts (auto-created if missing) ===
        base_dir_env = os.getenv("BASE_DIR", "artifacts")
        self.BASE_DIR = (
            base_dir_env
            if os.path.isabs(base_dir_env)
            else os.path.join(self.PROJECT_ROOT, base_dir_env)
        )

        self.RAW_DATA_DIR = os.path.join(self.BASE_DIR, "data", "raw")
        self.PROCESSED_DATA_DIR = os.path.join(self.BASE_DIR, "data", "processed")
        self.METADATA_DIR = os.path.join(self.BASE_DIR, "data", "metadata")
        self.VOCABULARY = os.path.join(self.BASE_DIR, "data", "vocabulary")
        self.LOG_DIR = os.path.join(self.BASE_DIR, "logs")

        # === Tokenizer Settings ===
        self._tokenizer_merges = 10000
        self._lowercase = False
        tokenizer_dir_env = os.getenv("TOKENIZER_OUTPUT_DIR", self.VOCABULARY)
        self._tokenizer_output_dir = self._resolve_path(tokenizer_dir_env)

        self._ensure_directories_exist()
        Config._is_initialized = True

    def _ensure_directories_exist(self):
        ensure_all_dirs_exist(
            [
                self.RAW_DATA_DIR,
                self.PROCESSED_DATA_DIR,
                self.METADATA_DIR,
                self.VOCABULARY,
                self.LOG_DIR,
            ]
        )

    def apply_cli_overrides(self, args):
        if _was_explicit(args, "debug"):
            print(f"[Config] Overriding 'debug' from CLI: {args.debug}")
            self.debug = args.debug

        if _was_explicit(args, "tokenizer_merges"):
            print(
                f"[Config] Overriding 'tokenizer_merges' from CLI: {self._tokenizer_merges} → {args.tokenizer_merges}"
            )
            self.tokenizer_merges = args.tokenizer_merges

        if _was_explicit(args, "lowercase"):
            print(
                f"[Config] Overriding 'lowercase' from CLI: {self._lowercase} → {args.lowercase}"
            )
            self.lowercase = args.lowercase

        if _was_explicit(args, "tokenizer_output_dir"):
            print(
                f"[Config] Overriding 'tokenizer_output_dir' from CLI: {self._tokenizer_output_dir} → {args.tokenizer_output_dir}"
            )
            self.tokenizer_output_dir = args.tokenizer_output_dir

    def load_from_yaml(self, path: str):
        """
        Override config values from a YAML config file.
        Logs changes to config values.
        """
        if not os.path.exists(path):
            print(f"[Config] YAML config file not found: {path}")
            return

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        print(f"[Config] Loaded YAML config: {path}")

        if "debug" in data:
            print(f"[Config] Overriding 'debug': {self._debug} → {data['debug']}")
            self.debug = data["debug"]

        if "tokenizer_merges" in data:
            print(
                f"[Config] Overriding 'tokenizer_merges': {self._tokenizer_merges} → {data['tokenizer_merges']}"
            )
            self.tokenizer_merges = data["tokenizer_merges"]

        if "lowercase" in data:
            print(
                f"[Config] Overriding 'lowercase': {self._lowercase} → {data['lowercase']}"
            )
            self.lowercase = data["lowercase"]

        if "tokenizer_output_dir" in data:
            print(
                f"[Config] Overriding 'tokenizer_output_dir': {self._tokenizer_output_dir} → {data['tokenizer_output_dir']}"
            )
            self.tokenizer_output_dir = data["tokenizer_output_dir"]

    @property
    def config_path(self):
        return self._config_path

    @config_path.setter
    def config_path(self, value):
        if not isinstance(value, str):
            raise ValueError("config_path must be a string.")
        self._config_path = value

    @property
    def debug(self):
        return self._debug

    @debug.setter
    def debug(self, value):
        if not isinstance(value, bool):
            raise ValueError("debug must be a boolean.")
        self._debug = value

    @property
    def tokenizer_merges(self) -> int:
        return self._tokenizer_merges

    @tokenizer_merges.setter
    def tokenizer_merges(self, value: int):
        if not isinstance(value, int):
            raise ValueError("tokenizer_merges must be an integer.")
        self._tokenizer_merges = value

    @property
    def lowercase(self) -> bool:
        return self._lowercase

    @lowercase.setter
    def lowercase(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("lowercase must be a boolean.")
        self._lowercase = value

    @property
    def tokenizer_output_dir(self) -> Optional[str]:
        return self._tokenizer_output_dir

    @tokenizer_output_dir.setter
    def tokenizer_output_dir(self, value: str):
        if not isinstance(value, str):
            raise ValueError("tokenizer_output_dir must be a string.")
        self._tokenizer_output_dir = self._resolve_path(value)

    def print_config_info(self):
        print("=" * 50)
        print("📂 Configured Dataset Directories")
        print("-" * 50)
        print(f"{'Datasets/raw dir:':25} {self.DATASETS_RAW_DIR}")
        print(f"{'Datasets/processed dir:':25} {self.DATASETS_PROCESSED_DIR}")
        print(f"{'Artifacts/raw dir:':25} {self.RAW_DATA_DIR}")
        print(f"{'Artifacts/processed dir:':25} {self.PROCESSED_DATA_DIR}")
        print(f"{'Vocab dir:':25} {self.VOCABULARY}")
        print("=" * 50)

    def _resolve_path(self, val: Optional[str]) -> Optional[str]:
        if not val:
            return None
        if os.path.isabs(val):
            return val
        # Tier 1: try resolving relative to BASE_DIR
        base_resolved = os.path.join(self.BASE_DIR, val)
        if os.path.exists(base_resolved):
            print(f"[Config] Resolved (BASE_DIR): {val} → {base_resolved}")
            return base_resolved
        # Tier 2: try resolving relative to PROJECT_ROOT
        root_resolved = os.path.join(self.PROJECT_ROOT, val)
        if os.path.exists(root_resolved):
            print(f"[Config] Resolved (PROJECT_ROOT): {val} → {root_resolved}")
            return root_resolved
        # Fallback: assume BASE_DIR anyway
        fallback = base_resolved
        print(f"[Config] Resolved (fallback to BASE_DIR): {val} → {fallback}")
        return fallback

    @classmethod
    def initialize(cls):
        if not cls._is_initialized:
            cls()

    @classmethod
    def is_initialized(cls):
        return cls._is_initialized

    @classmethod
    def reset(cls):
        cls._is_initialized = False
        cls._instances = {}
