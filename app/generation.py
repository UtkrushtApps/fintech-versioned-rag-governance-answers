from __future__ import annotations

import hashlib
import re

from anthropic import Anthropic
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.models import AnswerRequest, ApplicationContext, Citation


class LLMNotConfigured(RuntimeError):
    pass


class Generator:
    def __init__(self) -> None:
        self.settings = get_settings()

    def prompt_fingerprint(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode('utf-8')).hexdigest()

    def build_prompt(
        self,
        request: AnswerRequest,
        context: ApplicationContext,
        evidence_text: str,
        citations: list[Citation],
    ) -> str:
        citation_lines = ', '.join(c.citation_id for c in citations) or 'none'
        return (
            'You are a compliance assistant for loan disclosure reviewers. '
            'Answer only from the supplied evidence. Cite the evidence identifiers used.\n\n'
            f'Application: {context.application_id}\n'
            f'Product: {context.product_id}\n'
            f'Jurisdiction: {context.jurisdiction}\n'
            f'Signed at: {context.signed_at.isoformat()}\n'
            f'Available citations: {citation_lines}\n\n'
            f'Question: {request.question}\n\n'
            f'Evidence:\n{evidence_text}\n\n'
            'If the evidence is insufficient, say so clearly.'
        )

    @staticmethod
    def extract_citation_ids(answer: str) -> set[str]:
        # Expected format: C1, C2, ...
        return set(re.findall(r'\bC\d+\b', answer or ''))

    @retry(wait=wait_exponential(multiplier=0.5, min=0.5, max=4), stop=stop_after_attempt(2), reraise=True)
    def generate(self, prompt: str) -> str:
        provider = self.settings.llm_provider.lower().strip()
        if provider == 'anthropic':
            if not self.settings.anthropic_api_key:
                raise LLMNotConfigured('Anthropic provider key is not configured')
            client = Anthropic(api_key=self.settings.anthropic_api_key)
            message = client.messages.create(
                model=self.settings.anthropic_model,
                max_tokens=700,
                temperature=0,
                messages=[{'role': 'user', 'content': prompt}],
            )
            return ''.join(block.text for block in message.content if getattr(block, 'type', '') == 'text')

        if not self.settings.openai_api_key:
            raise LLMNotConfigured('OpenAI provider key is not configured')

        kwargs = {'api_key': self.settings.openai_api_key}
        if self.settings.openai_base_url:
            kwargs['base_url'] = self.settings.openai_base_url
        client = OpenAI(**kwargs)
        response = client.chat.completions.create(
            model=self.settings.openai_model,
            temperature=0,
            messages=[{'role': 'user', 'content': prompt}],
        )
        return response.choices[0].message.content or ''
