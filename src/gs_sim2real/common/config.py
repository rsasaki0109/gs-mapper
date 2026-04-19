"""Configuration loading and management.

This module handles loading YAML configuration files for datasets and
training hyperparameters, merging them with CLI overrides, and providing
a unified config object for the pipeline.

Expected config files:
- configs/datasets.yaml: Dataset metadata (URLs, descriptions, splits)
- configs/training.yaml: 3DGS training hyperparameters
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def get_project_root() -> Path:
    """Return the project root directory (where pyproject.toml lives)."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find project root (no pyproject.toml found in parents).")


def load_config(path: Path | str) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed configuration as a dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def load_datasets_config() -> dict[str, Any]:
    """Load the datasets configuration from configs/datasets.yaml.

    Returns:
        Dictionary of dataset configurations keyed by dataset name.
    """
    root = get_project_root()
    return load_config(root / "configs" / "datasets.yaml")


def load_training_config() -> dict[str, Any]:
    """Load the training configuration from configs/training.yaml.

    Returns:
        Dictionary of training hyperparameters.
    """
    root = get_project_root()
    return load_config(root / "configs" / "training.yaml")


def get_dataset_config(name: str) -> dict[str, Any]:
    """Get configuration for a specific dataset by name.

    Args:
        name: Dataset identifier (e.g. "covla", "mcd", "autoware_leo_drive_bagN").

    Returns:
        Dictionary of dataset configuration.

    Raises:
        ValueError: If the dataset name is not found in the configuration.
    """
    datasets = load_datasets_config()
    if name not in datasets:
        available = ", ".join(sorted(datasets.keys()))
        raise ValueError(f"Unknown dataset '{name}'. Available datasets: {available}")
    return datasets[name]
