from __future__ import annotations

import os
from pathlib import Path

import yaml

# Repository root (this file lives at <root>/llm_quant_bench/config.py).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv

    _HAS_DOTENV = True
except ImportError:  # python-dotenv is optional; raw env vars still work without it.
    _HAS_DOTENV = False

_ENV_LOADED = False


def _ensure_env_loaded() -> None:
    """Load a .env file from the project root once, if python-dotenv is installed."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    if _HAS_DOTENV:
        load_dotenv(PROJECT_ROOT / ".env")
    _ENV_LOADED = True


def get_api_key(env_var: str = "OPENROUTER_API_KEY") -> str:
    """Resolve an API key from an environment variable.

    Keys are read from the process environment, falling back to a ``.env`` file
    in the project root (see ``.env.example``). We never read keys from tracked
    config files, so nothing secret ends up in version control.
    """
    _ensure_env_loaded()
    key = os.environ.get(env_var)
    if not key:
        raise RuntimeError(
            f"API key not found: set the '{env_var}' environment variable, "
            f"or add `{env_var}=...` to a .env file in the project root "
            f"(copy .env.example to .env)."
        )
    return key


def load_config(config_path: str | Path) -> dict:
    """Load a provider/run config YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)
