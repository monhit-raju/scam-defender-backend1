from __future__ import annotations

import datetime as dt
from typing import Any

import requests


class CloudVerificationService:
    def __init__(self, endpoint_url: str = "", api_key: str = "", timeout_seconds: int = 8):
        self.endpoint_url = str(endpoint_url or "").strip()
        self.api_key = str(api_key or "").strip()
        self.timeout_seconds = max(1, int(timeout_seconds))

    @property
    def enabled(self) -> bool:
        return bool(self.endpoint_url)

    def submit(
        self,
        scan_type: str,
        payload: dict[str, Any],
        model_output: dict[str, Any],
        policy: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.enabled:
            return {
                "submitted": False,
                "status": "disabled",
                "timestamp": dt.datetime.utcnow().isoformat(),
                "message": "Cloud verification endpoint not configured.",
            }

        request_payload = {
            "scan_type": scan_type,
            "payload": payload,
            "model_output": model_output,
            "policy": policy,
            "timestamp": dt.datetime.utcnow().isoformat(),
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = requests.post(
                self.endpoint_url,
                json=request_payload,
                headers=headers,
                timeout=self.timeout_seconds,
            )
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                response_payload = response.json()
            else:
                response_payload = {"raw": response.text[:4000]}

            return {
                "submitted": True,
                "status": "ok" if response.ok else "failed",
                "http_status": int(response.status_code),
                "timestamp": dt.datetime.utcnow().isoformat(),
                "response": response_payload,
            }
        except Exception as exc:
            return {
                "submitted": True,
                "status": "error",
                "timestamp": dt.datetime.utcnow().isoformat(),
                "error": str(exc),
            }
