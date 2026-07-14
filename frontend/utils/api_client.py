"""
frontend/utils/api_client.py

Thin HTTP client wrapper the Streamlit app uses to talk to the FastAPI
backend. Centralizing this means every page reuses the same base URL,
timeout, and error handling.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000/api/v1")
TIMEOUT_SECONDS = 30


class ApiError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"[{status_code}] {detail}")


def _handle_response(resp: requests.Response) -> Dict[str, Any]:
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise ApiError(resp.status_code, detail)
    return resp.json()


def _server_root_url() -> str:
    """API_BASE_URL is e.g. http://localhost:8000/api/v1 -> strip to http://localhost:8000"""
    return API_BASE_URL.split("/api/")[0]


def get_health() -> Dict[str, Any]:
    resp = requests.get(f"{_server_root_url()}/health", timeout=TIMEOUT_SECONDS)
    return _handle_response(resp)


def predict(payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=TIMEOUT_SECONDS)
    return _handle_response(resp)


def batch_predict(payloads: list) -> Dict[str, Any]:
    resp = requests.post(f"{API_BASE_URL}/batch_predict", json={"requests": payloads}, timeout=TIMEOUT_SECONDS)
    return _handle_response(resp)


def get_model_metrics() -> Dict[str, Any]:
    resp = requests.get(f"{API_BASE_URL}/model_metrics", timeout=TIMEOUT_SECONDS)
    return _handle_response(resp)


def get_feature_importance() -> Dict[str, Any]:
    resp = requests.get(f"{API_BASE_URL}/feature_importance", timeout=TIMEOUT_SECONDS)
    return _handle_response(resp)


def get_prediction_history(limit: int = 50, offset: int = 0) -> Dict[str, Any]:
    resp = requests.get(
        f"{API_BASE_URL}/prediction_history", params={"limit": limit, "offset": offset}, timeout=TIMEOUT_SECONDS
    )
    return _handle_response(resp)


def get_model_information() -> Dict[str, Any]:
    resp = requests.get(f"{API_BASE_URL}/model_information", timeout=TIMEOUT_SECONDS)
    return _handle_response(resp)


def trigger_retrain(regenerate_dataset: bool = False, num_records: Optional[int] = None) -> Dict[str, Any]:
    body = {"regenerate_dataset": regenerate_dataset}
    if num_records:
        body["num_records"] = num_records
    resp = requests.post(f"{API_BASE_URL}/retrain", json=body, timeout=600)
    return _handle_response(resp)
