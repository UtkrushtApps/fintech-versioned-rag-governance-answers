from __future__ import annotations

import hashlib
import json
from typing import Any

import redis

from app.config import get_settings


class RetrievalCache:
    def __init__(self) -> None:
        self.client = redis.Redis.from_url(get_settings().redis_url, decode_responses=True)

    @staticmethod
    def _hash_context(context_fingerprint: str) -> str:
        return hashlib.sha256(context_fingerprint.encode('utf-8')).hexdigest()[:20]

    def get(self, *, application_id: str, question: str, context_fingerprint: str) -> dict[str, Any] | None:
        raw = self.client.get(self._key(application_id, question, context_fingerprint))
        if not raw:
            return None
        return json.loads(raw)

    def set(self, *, application_id: str, question: str, context_fingerprint: str, payload: dict[str, Any]) -> None:
        self.client.set(
            self._key(application_id, question, context_fingerprint),
            json.dumps(payload, default=str),
            ex=3600,
        )

    def _key(self, application_id: str, question: str, context_fingerprint: str) -> str:
        settings = get_settings()
        norm_question = question.strip().lower()
        ctx_hash = self._hash_context(context_fingerprint)
        return f"advisor:v1:answer:{settings.index_id}:{application_id}:{ctx_hash}:{norm_question}"
