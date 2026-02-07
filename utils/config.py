"""Configuration management for HelixForge.

Validates config.yaml on startup, provides sensible defaults,
and supports environment variable overrides (12-factor compliance).

Environment variable convention:
    HELIXFORGE_LLM__PROVIDER=openai  →  config["llm"]["provider"] = "openai"
    HELIXFORGE_API__PORT=9000        →  config["api"]["port"] = 9000
    Double underscore (__) separates nesting levels.
"""

import os
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

from utils.errors import ConfigurationError


# ------------------------------------------------------------------ #
#  Pydantic models for each config section                            #
# ------------------------------------------------------------------ #

class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-large"
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=4096, gt=0)


class ProcessingConfig(BaseModel):
    confidence_threshold: float = Field(default=0.80, ge=0, le=1)
    max_file_size_mb: int = Field(default=500, gt=0)
    temp_storage_path: str = "./data/temp/"


class FusionFileConfig(BaseModel):
    default_join_strategy: str = "auto"
    similarity_threshold: float = Field(default=0.85, ge=0, le=1)
    imputation_method: str = "mean"

    @field_validator("default_join_strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        allowed = {"auto", "exact_key", "semantic_similarity", "probabilistic", "temporal"}
        if v.lower() not in allowed:
            raise ValueError(f"Invalid join strategy: {v}. Must be one of {sorted(allowed)}")
        return v.lower()

    @field_validator("imputation_method")
    @classmethod
    def validate_imputation(cls, v: str) -> str:
        allowed = {"mean", "median", "mode", "knn", "model"}
        if v.lower() not in allowed:
            raise ValueError(f"Invalid imputation method: {v}. Must be one of {sorted(allowed)}")
        return v.lower()


class OutputConfig(BaseModel):
    artifact_dir: str = "./outputs/"


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = Field(default=8000, gt=0, le=65535)
    cors_origins: List[str] = Field(default_factory=lambda: ["http://localhost:3000"])


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "json"
    output: str = "stdout"

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            raise ValueError(f"Invalid log level: {v}. Must be one of {sorted(valid)}")
        return v.upper()

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        if v.lower() not in {"json", "text"}:
            raise ValueError(f"Invalid log format: {v}. Must be 'json' or 'text'")
        return v.lower()


class AppConfig(BaseModel):
    """Top-level application configuration with validation."""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    fusion: FusionFileConfig = Field(default_factory=FusionFileConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# ------------------------------------------------------------------ #
#  Environment variable overlay                                       #
# ------------------------------------------------------------------ #

_ENV_PREFIX = "HELIXFORGE_"


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply HELIXFORGE_* environment variable overrides.

    Convention: HELIXFORGE_SECTION__KEY=value
    Double underscore (__) separates nesting levels.
    """
    for key, value in os.environ.items():
        if not key.startswith(_ENV_PREFIX):
            continue
        # Skip HELIXFORGE_CONFIG (that's the config file path)
        if key == "HELIXFORGE_CONFIG":
            continue

        parts = key[len(_ENV_PREFIX):].lower().split("__")
        if len(parts) < 2:
            continue

        # Navigate to the right nested dict
        target = config
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]

        # Set the value (attempt type coercion for common types)
        raw = value
        if raw.lower() in ("true", "false"):
            target[parts[-1]] = raw.lower() == "true"
        elif raw.isdigit():
            target[parts[-1]] = int(raw)
        else:
            try:
                target[parts[-1]] = float(raw)
            except ValueError:
                target[parts[-1]] = raw

    return config


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #

# Cached config singleton
_cached_config: Optional[Dict[str, Any]] = None


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load and validate configuration.

    Order of precedence (highest wins):
    1. Environment variables (HELIXFORGE_*)
    2. Config file (config.yaml)
    3. Built-in defaults

    Args:
        config_path: Path to config file. Defaults to HELIXFORGE_CONFIG
                     env var, then "config.yaml".

    Returns:
        Validated configuration dictionary.

    Raises:
        ConfigurationError: If config file is malformed or values are invalid.
    """
    global _cached_config
    if _cached_config is not None:
        return _cached_config

    if config_path is None:
        config_path = os.environ.get("HELIXFORGE_CONFIG", "config.yaml")

    # Load YAML
    raw: Dict[str, Any] = {}
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                raw = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Failed to parse config file: {config_path}",
                detail=str(e),
            )

    if not isinstance(raw, dict):
        raise ConfigurationError(
            f"Config file must contain a YAML mapping, got {type(raw).__name__}",
        )

    # Apply env overrides
    raw = _apply_env_overrides(raw)

    # Validate
    validate_config(raw)

    _cached_config = raw
    return _cached_config


def validate_config(raw: Dict[str, Any]) -> AppConfig:
    """Validate a raw config dict against the schema.

    Returns:
        Validated AppConfig (also used for its side effect of raising
        ConfigurationError on invalid input).

    Raises:
        ConfigurationError: With human-readable message on failure.
    """
    try:
        return AppConfig(**raw)
    except Exception as e:
        raise ConfigurationError(
            "Invalid configuration",
            detail=str(e),
        )


def reset_config() -> None:
    """Reset the cached config (for testing)."""
    global _cached_config
    _cached_config = None
