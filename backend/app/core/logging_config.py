"""
backend/app/core/logging_config.py

Centralized logging setup using loguru. All backend/ML modules should do:
    from backend.app.core.logging_config import get_logger
    logger = get_logger(__name__)
"""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from backend.app.core.config import settings, PROJECT_ROOT

_LOG_DIR = PROJECT_ROOT / settings.paths.get("logs_dir", "logs")
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / "returnguard.log"

_CONFIGURED = False


def configure_logging() -> None:
    """Idempotently configure loguru sinks (console + rotating file)."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_cfg = settings.logging_config
    level = log_cfg.get("level", "INFO")
    rotation = log_cfg.get("rotation", "10 MB")
    retention = log_cfg.get("retention", "14 days")
    fmt = log_cfg.get(
        "format",
        "{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}",
    )

    logger.remove()  # remove default handler
    logger.add(sys.stderr, level=level, format=fmt, colorize=True)
    logger.add(
        _LOG_FILE,
        level=level,
        format=fmt,
        rotation=rotation,
        retention=retention,
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )
    _CONFIGURED = True


def get_logger(name: str = "returnguard"):
    """Return a bound loguru logger tagged with the calling module's name."""
    configure_logging()
    return logger.bind(module=name)
