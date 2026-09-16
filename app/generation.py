from __future__ import annotations

import hashlib

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

    def build_prompt(self, request: AnswerRequest, context: ApplicationContext, evidence_text: str,
                     citations: list[Citation], knowledge=None, missing_roles: set[str] | None = None) -> str:
        citation_lines = ', '.join(c.citation_id for c in citations) or 'none'
        applies = ''
        conflicts = ''
        if knowledge is not None:
            applies = '\n'.join(
                f"- {d['role']} ({d['jurisdiction']}) {d['document_id']} revision {d.get('revision_id')}"
                for d in knowledge.documents
            ) or '- none'
            if knowledge.conflicts:
                conflicts = '\n'.join(f'- {c.kind}: {c.detail}' for c in knowledge.conflicts)
        missing = ', '.join(sorted(missing_roles or set())) or 'none'
        return (
            'You are a compliance assistant for loan disclosure reviewers. '
            'Answer only from the supplied evidence, and only about the documents listed as applying '
            'to this application. State the figure that governs this borrower, and say which document '
            'each part of the answer comes from. If the records below disagree, say so plainly and do '
            'not choose between them. If a required document is missing from the evidence, say the '
            'answer is incomplete rather than filling the gap.\n\n'
            f'Application: {context.application_id}\n'
            f'Product: {context.product_id}\n'
            f'Jurisdiction: {context.jurisdiction}\n'
            f'Signed at: {context.signed_at}\n\n'
            f'Documents that apply:\n{applies}\n\n'
            + (f'Disagreements in the bank\'s records:\n{conflicts}\n\n' if conflicts else '')
            + f'Required document types with no evidence supplied: {missing}\n\n'
            f'Question: {request.question}\n\n'
            f'Evidence:\n{evidence_text}\n\n'
            f'Cite using these identifiers: {citation_lines}\n'
        )

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
