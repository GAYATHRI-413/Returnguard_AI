"""
backend/app/core/config.py

Central application settings loader.

Combines:
  - config/config.yaml   -> non-secret defaults (paths, hyperparameters, thresholds)
  - config/.env / env vars -> secrets and deployment-specific overrides

Everywhere else in the backend should do:
    from backend.app.core.config import settings
instead of re-reading files.
"""
from __future__ import annotations

import os
import yaml
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = two levels above this file (backend/app/core/config.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_YAML_PATH = PROJECT_ROOT / "config" / "config.yaml"
ENV_FILE_PATH = PROJECT_ROOT / ".env"


def _load_yaml_config() -> Dict[str, Any]:
    """Load config/config.yaml into a plain dict. Returns {} if missing."""
    if not CONFIG_YAML_PATH.exists():
        return {}
    with open(CONFIG_YAML_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class Settings(BaseSettings):
    """
    Typed settings object. Secret / environment-specific fields are declared
    explicitly (pulled from .env or real env vars). Everything else (paths,
    hyperparameters, thresholds) is loaded dynamically from config.yaml and
    exposed via the `yaml_config` dict plus convenience properties below.
    """

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE_PATH),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Secrets / deployment-specific (from .env) ----
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    DATABASE_URL: Optional[str] = None

    JWT_SECRET_KEY: str = "dev-only-insecure-secret-change-me"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_MINUTES: int = 60

    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD: str = "admin123"

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    API_BASE_URL: str = "http://localhost:8000/api/v1"
    STREAMLIT_SERVER_PORT: int = 8501

    PORT: Optional[int] = None  # Render sets this automatically
    RENDER: bool = False

    # ---- Loaded from config.yaml (not env-backed) ----
    yaml_config: Dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "yaml_config", _load_yaml_config())

    # ------------------------------------------------------------------
    # Convenience accessors over yaml_config
    # ------------------------------------------------------------------
    @property
    def app_name(self) -> str:
        return self.yaml_config.get("app", {}).get("name", "ReturnGuard AI")

    @property
    def app_version(self) -> str:
        return self.yaml_config.get("app", {}).get("version", "1.0.0")

    @property
    def api_prefix(self) -> str:
        return self.yaml_config.get("api", {}).get("prefix", "/api/v1")

    @property
    def cors_origins(self) -> List[str]:
        return self.yaml_config.get("api", {}).get("cors_origins", ["*"])

    @property
    def paths(self) -> Dict[str, str]:
        return self.yaml_config.get("paths", {})

    def resolved_path(self, key: str) -> Path:
        """Return an absolute Path for a given key in config.yaml -> paths."""
        rel = self.paths.get(key)
        if rel is None:
            raise KeyError(f"Path key '{key}' not found in config.yaml paths section")
        return PROJECT_ROOT / rel

    @property
    def dataset_config(self) -> Dict[str, Any]:
        return self.yaml_config.get("dataset", {})

    @property
    def feature_config(self) -> Dict[str, Any]:
        return self.yaml_config.get("features", {})

    @property
    def training_config(self) -> Dict[str, Any]:
        return self.yaml_config.get("training", {})

    @property
    def risk_scoring_config(self) -> Dict[str, Any]:
        return self.yaml_config.get("risk_scoring", {})

    @property
    def explainability_config(self) -> Dict[str, Any]:
        return self.yaml_config.get("explainability", {})

    @property
    def logging_config(self) -> Dict[str, Any]:
        return self.yaml_config.get("logging", {})

    @property
    def effective_database_url(self) -> str:
        """
        Returns a usable SQLAlchemy DB URL.
        Priority: DATABASE_URL env var (Postgres in prod) -> SQLite fallback.
        """
        if self.DATABASE_URL:
            return self.DATABASE_URL
        sqlite_rel_path = self.yaml_config.get("database", {}).get(
            "sqlite_fallback_path", "database/returnguard.db"
        )
        sqlite_path = PROJECT_ROOT / sqlite_rel_path
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{sqlite_path}"

    @property
    def effective_port(self) -> int:
        """Render injects PORT env var; prefer it if present."""
        return self.PORT or self.API_PORT


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
