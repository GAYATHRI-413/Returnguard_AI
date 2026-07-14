"""
backend/app/db/database.py

SQLAlchemy engine/session setup. Uses Postgres if DATABASE_URL is set,
otherwise falls back to a local SQLite file automatically.
"""
from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from backend.app.core.config import settings
from backend.app.core.logging_config import get_logger

logger = get_logger(__name__)

DATABASE_URL = settings.effective_database_url

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    echo=settings.yaml_config.get("database", {}).get("echo_sql", False),
    connect_args=connect_args,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI dependency that yields a DB session and closes it afterwards."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables if they don't already exist."""
    from backend.app.db import models  # noqa: F401  (ensures models are registered)

    Base.metadata.create_all(bind=engine)
    logger.info(f"Database initialized at: {DATABASE_URL}")
