from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import Settings


@dataclass(slots=True)
class HostedLLMClient:
    settings: Settings

    @property
    def enabled(self) -> bool:
        return bool(self.settings.llm_base_url and self.settings.llm_api_key and self.settings.llm_model)

    def _post(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        if not self.enabled:
            return None

        headers = {
            "Authorization": f"Bearer {self.settings.llm_api_key}",
            "Content-Type": "application/json",
        }
        url = self.settings.llm_base_url.rstrip("/") + "/chat/completions"
        try:
            with httpx.Client(timeout=self.settings.request_timeout_seconds) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
        except httpx.HTTPError:
            return None

        try:
            return response.json()
        except json.JSONDecodeError:
            return None

    def extract_state(self, messages: list[dict[str, str]]) -> dict[str, Any] | None:
        payload = {
            "model": self.settings.llm_model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You extract structured hiring constraints for a grounded SHL assessment recommender. "
                        "Return strict JSON with keys: use_case, job_level, languages, must_have, avoid, "
                        "needs_clarification, clarification_focus, comparison_targets."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(messages, ensure_ascii=True),
                },
            ],
        }
        data = self._post(payload)
        if not data:
            return None
        try:
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            return None

    def write_reply(self, prompt: str) -> str | None:
        payload = {
            "model": self.settings.llm_model,
            "temperature": self.settings.llm_temperature,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Write concise grounded replies for an SHL assessment recommender. "
                        "Use only the evidence provided. Do not invent products or URLs. "
                        "No markdown tables."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        data = self._post(payload)
        if not data:
            return None
        try:
            return str(data["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError):
            return None
