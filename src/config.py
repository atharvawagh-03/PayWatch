"""
Configuration loader for PayWatch
"""

from pathlib import Path
from typing import Any, Dict, List
import yaml


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = BASE_DIR / "config.yaml"


class Config:
    def __init__(self, config_dict: Dict[str, Any]):
        self.raw = config_dict
        self.version: str = config_dict.get("version", "1.0.0")
        self.seed: int = config_dict.get("seed", 42)

        self.generator: Dict[str, Any] = config_dict.get("generator", {})
        self.features: Dict[str, Any] = config_dict.get("features", {})
        self.iqr: Dict[str, Any] = config_dict.get("iqr", {})
        self.isolation_forest: Dict[str, Any] = config_dict.get("isolation_forest", {})
        self.timeseries: Dict[str, Any] = config_dict.get("timeseries", {})
        self.ensemble: Dict[str, Any] = config_dict.get("ensemble", {})
        self.api: Dict[str, Any] = config_dict.get("api", {})

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Config":
        config_path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(data or {})


def get_config(path: Path | str | None = None) -> Config:
    return Config.load(path)
