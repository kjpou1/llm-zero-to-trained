# === Project bootstrap ===
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../..")))
from utils.env_setup import setup_src_path

setup_src_path()
# =========================
