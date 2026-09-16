from __future__ import annotations

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
from app.knowledge import KnowledgeResolver
from app.models import AnswerRequest, AnswerResponse, ProvenanceRecord

logger = logging.getLogger(__name__)
app = FastAPI(title='Fintech Disclosure Advisor')

repo = GraphRepository()
resolver = KnowledgeResolver(repo)
retriever = HybridRetriever()
expander = GraphExpander()
assembler = ContextAssembler()
generator = Generator()
audit = AuditLogger()
cache = RetrievalCache()


@app.get('/health')
def health() -> dict[str, str]:
    return {'status': 'ok'}


@app.post('/v1/advisor/answer', response_model=AnswerResponse)
def answer(request: AnswerRequest) -> AnswerResponse:
    request_id = str(uuid.uuid4())
    try:
        context = repo.load_application(request.application_id)
        knowledge = resolver.resolve(context)
        fingerprint = knowledge.fingerprint()

        cached = cache.get(request.application_id, request.question, fingerprint)
        if cached and cached.get('answer'):
            return AnswerResponse(**cached)

        conflict_warnings = [f'{c.kind}: {c.detail} (sources: {", ".join(c.sources)})'
                             for c in knowledge.conflicts]

        # A product the bank never offered in this state has no answer, however close
        # the surviving disclosure text reads. Decline, and say what was checked.
        if not knowledge.offered:
            graph = expander.expand(context, [], knowledge)
            response = AnswerResponse(
                request_id=request_id,
                answer=(
                    f'No disclosure applies to {request.application_id}: '
                    f'{knowledge.offering_note} The bank\'s records hold no {context.product_id} '
                    f'disclosure in force for a {context.jurisdiction} borrower, so this question '
                    f'cannot be answered from them. Refer the application for review.'
                ),
                citations=[],
                provenance=ProvenanceRecord(
                    request_id=request_id,
                    application_id=request.application_id,
                    index_id=generator.settings.index_id,
                    chunk_ids=[],
                    graph_refs=[node.node_id for node in graph.nodes],
                    revision_ids=[],
                    manifest_refs=[generator.settings.index_id],
                ),
                warnings=['declined: no offering on record'] + conflict_warnings,
            )
            audit.record_retrieval(request_id, request.application_id, request.question, [], [], graph)
            cache.set(request.application_id, request.question, fingerprint, response.model_dump(mode='json'))
            return response

        candidates = retriever.retrieve(request.question, context, knowledge)
        missing = retriever.coverage(candidates, knowledge)
        graph = expander.expand(context, candidates, knowledge)
        evidence_text, citations, provenance = assembler.assemble(
            request_id, request.application_id, candidates, graph
        )
        audit.record_retrieval(request_id, request.application_id, request.question,
                               candidates, candidates, graph)

        warnings = list(conflict_warnings)
        if missing:
            warnings.append('incomplete evidence: no passage from ' + ', '.join(sorted(missing)))

        prompt = generator.build_prompt(request, context, evidence_text, citations,
                                        knowledge=knowledge, missing_roles=missing)
        answer_text = generator.generate(prompt)
        response = AnswerResponse(request_id=request_id, answer=answer_text, citations=citations,
                                  provenance=provenance, warnings=warnings)
        audit.record_generation(
            request_id,
            request.application_id,
            generator.settings.llm_provider,
            generator.settings.anthropic_model if generator.settings.llm_provider == 'anthropic' else generator.settings.openai_model,
            prompt,
            answer_text,
            citations,
            provenance,
        )
        cache.set(request.application_id, request.question, fingerprint, response.model_dump(mode='json'))
        return response
    except LLMNotConfigured as exc:
        raise HTTPException(status_code=424, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception('answer generation failed')
        raise HTTPException(status_code=500, detail='answer generation failed') from exc
