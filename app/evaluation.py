from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from psycopg.types.json import Json

from app.config import BASE_DIR, get_settings
from app.context_assembly import ContextAssembler
from app.generation import Generator, LLMNotConfigured
from app.graph_expansion import GraphExpander
from app.graph_repository import GraphRepository, get_conn
from app.hybrid_retrieval import HybridRetriever
from app.models import DimensionResult, EvaluationCase, EvaluationReport

EVAL_PATH = BASE_DIR / 'data' / 'eval_queries.jsonl'


def load_eval_cases(path: Path = EVAL_PATH) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                cases.append(EvaluationCase(**json.loads(line)))
    return cases


class Evaluator:
    def __init__(self) -> None:
        self.repo = GraphRepository()
        self.retriever = HybridRetriever()
        self.expander = GraphExpander()
        self.assembler = ContextAssembler()
        self.generator = Generator()

    def evaluate(self, run_label: str = 'local') -> EvaluationReport:
        dimensions: list[DimensionResult] = []
        for case in load_eval_cases():
            context = self.repo.load_application(case.application_id)
            chunks = self.retriever.retrieve(case.question, context)
            graph = self.expander.expand(context, chunks)
            evidence_text, citations, provenance = self.assembler.assemble(
                request_id='eval',
                application_id=case.application_id,
                chunks=chunks,
                graph=graph,
            )

            answer_text: str | None = None
            generation_error: str | None = None
            try:
                # Faithfulness is about the generated answer; only attempt when provider keys exist.
                provider = self.generator.settings.llm_provider.lower().strip()
                if provider == 'anthropic' and self.generator.settings.anthropic_api_key:
                    prompt = self.generator.build_prompt(
                        request=type('Req', (), {'question': case.question, 'application_id': case.application_id})(),
                        context=context,
                        evidence_text=evidence_text,
                        citations=citations,
                    )
                    answer_text = self.generator.generate(prompt)
                elif provider != 'anthropic' and self.generator.settings.openai_api_key:
                    prompt = self.generator.build_prompt(
                        request=type('Req', (), {'question': case.question, 'application_id': case.application_id})(),
                        context=context,
                        evidence_text=evidence_text,
                        citations=citations,
                    )
                    answer_text = self.generator.generate(prompt)
            except LLMNotConfigured as exc:
                generation_error = str(exc)
            except Exception as exc:
                generation_error = f'{type(exc).__name__}: {exc}'

            dimensions.extend(
                self._score_case(
                    case=case,
                    context=context,
                    question=case.question,
                    retrieved_chunks=chunks,
                    graph=graph,
                    citations=citations,
                    provenance=provenance,
                    answer_text=answer_text,
                    generation_error=generation_error,
                )
            )

        report = EvaluationReport(run_label=run_label, dimensions=dimensions)
        self._persist(report)
        return report

    def _score_case(
        self,
        *,
        case: EvaluationCase,
        context: Any,
        question: str,
        retrieved_chunks: list[Any],
        graph: Any,
        citations: list[Any],
        provenance: Any,
        answer_text: str | None,
        generation_error: str | None,
    ) -> list[DimensionResult]:
        # Authorized evidence scope at signing.
        authorized = {
            r['revision_id']
            for r in self.repo.disclosure_revisions_for_context(
                product_id=context.product_id,
                jurisdiction=context.jurisdiction,
                signed_at=context.signed_at,
                limit=5,
            )
            if r.get('revision_id')
        }

        retrieved_revision_ids = {getattr(chunk, 'revision_id', None) for chunk in retrieved_chunks if getattr(chunk, 'revision_id', None)}

        # 1) Retrieval scope (version/jurisdiction/product/lifecycle)
        retrieval_scope_passed = bool(retrieved_chunks) and retrieved_revision_ids.issubset(authorized) and bool(
            retrieved_revision_ids & authorized
        )
        dims: list[DimensionResult] = [
            DimensionResult(
                name='retrieval_scope',
                passed=retrieval_scope_passed,
                detail={
                    'case_id': case.case_id,
                    'authorized_revision_ids': sorted(authorized),
                    'retrieved_revision_ids': sorted(retrieved_revision_ids),
                },
            )
        ]

        # 2) Graph attribution
        expected_revision = self.repo.disclosure_revision_for_context(
            product_id=context.product_id,
            jurisdiction=context.jurisdiction,
            signed_at=context.signed_at,
        )
        expected_revision_node_id = (
            self.repo.revision_node(expected_revision['revision_id'], signed_at=context.signed_at)['node_id']
            if expected_revision
            else None
        )
        product_node_id = self.repo.product_node(context.product_id, signed_at=context.signed_at)['node_id']

        graph_node_ids = {node.node_id for node in (getattr(graph, 'nodes', []) or [])}
        graph_attribution_passed = bool(graph_node_ids) and (product_node_id in graph_node_ids) and (
            expected_revision_node_id in graph_node_ids if expected_revision_node_id else True
        )

        dims.append(
            DimensionResult(
                name='graph_attribution',
                passed=graph_attribution_passed,
                detail={
                    'case_id': case.case_id,
                    'expected_revision_id': expected_revision['revision_id'] if expected_revision else None,
                    'graph_node_ids': sorted(graph_node_ids),
                },
            )
        )

        # 3) Citation support / grounding
        expected_citation_ids = {c.citation_id for c in citations}
        citation_ids_in_provenance = set()
        citation_supported = True
        for c in citations:
            if not c.chunk_id or not c.revision_id or not c.source_uri:
                citation_supported = False
            if c.revision_id and c.revision_id not in authorized:
                citation_supported = False
            citation_ids_in_provenance.add(c.citation_id)

        dims.append(
            DimensionResult(
                name='citation_grounding',
                passed=bool(expected_citation_ids) and citation_supported,
                detail={
                    'case_id': case.case_id,
                    'citation_ids': sorted(expected_citation_ids),
                    'authorized_revision_ids': sorted(authorized),
                },
            )
        )

        # 4) Answer faithfulness (deterministic heuristics; attempts generation only when keys exist)
        if answer_text is None:
            passed = False
            detail = {
                'case_id': case.case_id,
                'reason': generation_error or 'llm_not_configured',
            }
        else:
            # Heuristic: at least one expected citation id must appear.
            found = set(re.findall(r'\bC\d+\b', answer_text or ''))
            passed = bool(expected_citation_ids & found) and (bool(retrieved_chunks) or not expected_citation_ids)
            detail = {
                'case_id': case.case_id,
                'expected_citation_ids': sorted(expected_citation_ids),
                'found_citation_ids': sorted(found),
            }

        dims.append(
            DimensionResult(
                name='answer_faithfulness',
                passed=passed,
                detail=detail,
            )
        )

        return dims

    def _persist(self, report: EvaluationReport) -> None:
        settings = get_settings()
        with get_conn() as conn:
            run = conn.execute(
                """
                INSERT INTO evaluation_runs(run_label, corpus_id, index_id, code_fingerprint, completed_at, result_payload)
                VALUES (%s, %s, %s, %s, now(), %s)
                RETURNING run_id
                """,
                (report.run_label, settings.corpus_id, settings.index_id, 'starter-hardened', Json(report.model_dump(mode='json'))),
            ).fetchone()
            for dim in report.dimensions:
                conn.execute(
                    """
                    INSERT INTO quality_dimensions(run_id, case_id, dimension_name, observed_value, passed)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (run['run_id'], dim.detail.get('case_id', 'unknown'), dim.name, Json(dim.detail), dim.passed),
                )
            conn.commit()


def main() -> None:
    report = Evaluator().evaluate('manual')
    print(report.model_dump_json(indent=2))


if __name__ == '__main__':
    main()
