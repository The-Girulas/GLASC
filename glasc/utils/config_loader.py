"""
GLASC: Global Leverage & Asset Strategy Controller
Module: Config Loader

Charge la configuration depuis config.yaml
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel

class LLMConfig(BaseModel):
    provider: str
    model_path: str
    temperature: float = 0.1
    max_tokens: int = 512

class SystemConfig(BaseModel):
    log_level: str = "INFO"
    device: str = "auto"

class AppConfig(BaseModel):
    llm: LLMConfig
    system: SystemConfig

_CONFIG_CACHE: Optional[AppConfig] = None

def load_config(config_path: str = "config.yaml") -> AppConfig:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    # Locate config file (root of project)
    # Assuming this code runs from project root or installed package
    path = Path(config_path)
    if not path.exists():
        # Try to find it relative to this file if run from elsewhere
        path = Path(__file__).parent.parent.parent / config_path
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path.absolute()}")

    with open(path, "r") as f:
        raw_config = yaml.safe_load(f)

    _CONFIG_CACHE = AppConfig(**raw_config)
    return _CONFIG_CACHE
