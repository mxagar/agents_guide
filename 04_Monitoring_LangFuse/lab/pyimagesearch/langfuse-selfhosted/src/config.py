"""
Configuration loader for YAML config files.
Provides centralized access to application settings.
"""

import yaml
import os
from pathlib import Path


def load_config(path: str = None) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        path: Path to config file. If None, uses default configs/config.yaml
        
    Returns:
        Dictionary containing configuration settings
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    if path is None:
        # Default to configs/config.yaml relative to project root
        project_root = Path(__file__).parent.parent
        path = project_root / "configs" / "config.yaml"
    
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_llm_config() -> dict:
    """Get LLM-specific configuration"""
    config = load_config()
    return config.get("llm", {})


def get_langfuse_config() -> dict:
    """Get Langfuse-specific configuration"""
    config = load_config()
    return config.get("langfuse", {})


def get_evaluation_config() -> dict:
    """Get evaluation-specific configuration"""
    config = load_config()
    return config.get("evaluation", {})
