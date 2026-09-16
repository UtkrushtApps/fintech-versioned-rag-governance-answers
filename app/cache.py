from __future__ import annotations

import json
from typing import Any

import redis

from app.config import get_settings


class RetrievalCache:
    def __init__(self) -> None:
        self.client = redis.Redis.from_url(get_settings().redis_url, decode_responses=True)

    def get(self, application_id: str, question: str, knowledge_fingerprint: str) -> dict[str, Any] | None:
        raw = self.client.get(self._key(application_id, question, knowledge_fingerprint))
        if not raw:
            return None
        return json.loads(raw)

    def set(self, application_id: str, question: str, knowledge_fingerprint: str, payload: dict[str, Any]) -> None:
        self.client.set(
            self._key(application_id, question, knowledge_fingerprint),
            json.dumps(payload, default=str),
            ex=3600,
        )

    def _key(self, application_id: str, question: str, knowledge_fingerprint: str) -> str:
        settings = get_settings()
        return (
            f'advisor:answer:{settings.index_id}:{application_id}:'
            f'{knowledge_fingerprint}:{question.strip().lower()}'
        )
