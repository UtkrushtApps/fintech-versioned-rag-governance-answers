from __future__ import annotations

import hashlib
import logging
import uuid

from fastapi import FastAPI, HTTPException

from app.audit import AuditLogger
from app.cache import RetrievalCache
from app.context_assembly import ContextAssembler
from app.generation import Generator, LLMNotConfigured
from app.graph_expansion import GraphExpander
from app.graph_repository import GraphRepository
from app.hybrid_retrieval import HybridRetriever
from app.models import AnswerRequest, AnswerResponse

logger = logging.getLogger(__name__)
app = FastAPI(title='Fintech Disclosure Advisor')

repo = GraphRepository()
retriever = HybridRetriever()
expander = GraphExpander()
assembler = ContextAssembler()
generator = Generator()
audit = AuditLogger()
cache = RetrievalCache()


def context_fingerprint(context) -> str:
    # Include the signed snapshot to prevent cache leakage when the underlying application record changes.
    raw = f"{context.application_id}|{context.product_id}|{context.jurisdiction}|{context.signed_at.isoformat()}"
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


@app.get('/health')
def health() -> dict[str, str]:
    return {'status': 'ok'}


@app.post('/v1/advisor/answer', response_model=AnswerResponse)
def answer(request: AnswerRequest) -> AnswerResponse:
    request_id = str(uuid.uuid4())
    try:
        context = repo.load_application(request.application_id)
        ctx_fp = context_fingerprint(context)

        cached = cache.get(application_id=request.application_id, question=request.question, context_fingerprint=ctx_fp)
        if cached and cached.get('answer'):
            # Cached responses still include provenance/citations from prior generation.
            return AnswerResponse(**cached)

        candidates = retriever.retrieve(request.question, context)
        graph = expander.expand(context, candidates)

        evidence_text, citations, provenance = assembler.assemble(
            request_id=request_id,
            application_id=request.application_id,
            chunks=candidates,
            graph=graph,
        )

        audit.record_retrieval(
            request_id=request_id,
            application_id=request.application_id,
            query_text=request.question,
            candidates=candidates,
            selected=candidates[:4],
            graph=graph,
        )

        prompt = generator.build_prompt(request, context, evidence_text, citations)
        answer_text = generator.generate(prompt)

        warnings: list[str] = []
        expected_citation_ids = {c.citation_id for c in citations}
        found_citation_ids = generator.extract_citation_ids(answer_text)
        if expected_citation_ids and not (expected_citation_ids & found_citation_ids):
            warnings.append('Answer did not include any expected citation identifiers (C1/C2/...).')
        if citations and len(found_citation_ids) == 1 and len(expected_citation_ids) > 1:
            warnings.append('Answer cited fewer evidence identifiers than available; verify completeness.')

        response = AnswerResponse(
            request_id=request_id,
            answer=answer_text,
            citations=citations,
            provenance=provenance,
            warnings=warnings,
        )

        provider_model = (
            generator.settings.anthropic_model
            if generator.settings.llm_provider.lower().strip() == 'anthropic'
            else generator.settings.openai_model
        )

        audit.record_generation(
            request_id=request_id,
            application_id=request.application_id,
            provider=generator.settings.llm_provider,
            model_name=provider_model,
            prompt=prompt,
            answer=answer_text,
            citations=citations,
            provenance=provenance,
        )

        cache.set(
            application_id=request.application_id,
            question=request.question,
            context_fingerprint=ctx_fp,
            payload=response.model_dump(mode='json'),
        )
        return response

    except LLMNotConfigured as exc:
        raise HTTPException(status_code=424, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception('answer generation failed')
        raise HTTPException(status_code=500, detail='answer generation failed') from exc
