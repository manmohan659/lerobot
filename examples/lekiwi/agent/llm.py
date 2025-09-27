import json
import os
from typing import Any

import requests

from .types import Intent


PROMPT = (
    "You are a robotics assistant. Extract a JSON object with keys: task, object, constraints (list), confirm (bool).\n"
    "Examples: 'fetch me tissue' -> {\"task\":\"fetch\",\"object\":\"tissue\",\"constraints\":[],\"confirm\":true}.\n"
    "Only output valid JSON."
)


class IntentParser:
    def __init__(self) -> None:
        self.api_url = os.environ.get("LLM_API_URL", "")
        self.api_key = os.environ.get("LLM_API_KEY", "")

    def parse_intent(self, text: str) -> Intent:
        # Fallback simple rules if no API configured
        if not self.api_url or not self.api_key:
            if "tissue" in text.lower():
                return Intent(task="fetch", object="tissue")
            return Intent(task="fetch", object="object")

        payload = {
            "model": os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            "messages": [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": text},
            ],
            "temperature": 0.0,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        resp = requests.post(self.api_url, data=json.dumps(payload), headers=headers, timeout=10.0)
        resp.raise_for_status()
        content = resp.json()
        # Expect content["choices"][0]["message"]["content"] to be JSON
        try:
            msg = content["choices"][0]["message"]["content"]
            data = json.loads(msg)
            return Intent(
                task=str(data.get("task", "fetch")),
                object=str(data.get("object", "object")),
                constraints=tuple(data.get("constraints", [])),
                confirm=bool(data.get("confirm", True)),
            )
        except Exception:
            return Intent(task="fetch", object="tissue")


