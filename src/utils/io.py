"""I/O utilities for config loading and data paths."""

from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"


def load_config() -> dict:
    """Load the project config.yaml."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_raw_dir() -> Path:
    cfg = load_config()
    p = ROOT_DIR / cfg["data"]["raw_dir"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_processed_dir() -> Path:
    cfg = load_config()
    p = ROOT_DIR / cfg["data"]["processed_dir"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def season_range(cfg: dict) -> list[int]:
    """Return list of season start years from config."""
    return list(range(cfg["seasons"]["start"], cfg["seasons"]["end"]))
