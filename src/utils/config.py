"""
Configuration utilities for the Student Achievement Classification System.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent

def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = get_project_root() / "config" / "pipeline_config.yaml"

    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_data_path(filename: str, data_type: str = "raw") -> Path:
    """Get path to data file."""
    return get_project_root() / "data" / data_type / filename

def get_results_path(filename: str, result_type: str = "evaluation") -> Path:
    """Get path to results file."""
    return get_project_root() / "results" / result_type / filename

def ensure_dir(path: Path) -> None:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)