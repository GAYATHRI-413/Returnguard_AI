"""
backend/app/core/security.py

Password hashing + JWT issuing/verification for the Streamlit login screen
and any protected API routes (e.g. /retrain).
"""
from __future__ import annotations

import datetime as dt
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from backend.app.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain_password: str) -> str:
    return pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(subject: str, expires_minutes: Optional[int] = None) -> str:
    """Create a signed JWT for `subject` (typically the username)."""
    expire_minutes = expires_minutes or settings.JWT_EXPIRY_MINUTES
    expire = dt.datetime.utcnow() + dt.timedelta(minutes=expire_minutes)
    to_encode = {"sub": subject, "exp": expire, "iat": dt.datetime.utcnow()}
    return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> Optional[str]:
    """Return the subject (username) if the token is valid, else None."""
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None


def authenticate_admin(username: str, password: str) -> bool:
    """Simple single-admin auth backed by env vars (see .env.example)."""
    return username == settings.ADMIN_USERNAME and password == settings.ADMIN_PASSWORD
