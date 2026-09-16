from __future__ import annotations

import hashlib
import json
from pathlib import Path

from psycopg.types.json import Json

from app.config import BASE_DIR, get_settings
from app.graph_repository import GraphRepository, get_conn
from app.hybrid_retrieval import HybridRetriever
from app.knowledge import KnowledgeResolver
from app.models import DimensionResult, EvaluationCase, EvaluationReport

EVAL_PATH = BASE_DIR / 'data' / 'eval_queries.jsonl'

DIMENSIONS = (
    'evidence_completeness',   # every document type that applies is represented
    'scope_isolation',         # nothing from a document that does not apply
    'abstention_correctness',  # a combination never offered yields no evidence
    'conflict_visibility',     # disagreements in the records are reported, not resolved
    'citation_grounding',      # every passage used can be cited back to its source
)


def load_eval_cases(path: Path = EVAL_PATH) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                cases.append(EvaluationCase(case_id=row['case_id'], application_id=row['application_id'],
                                            question=row['question'], notes=row.get('notes')))
    return sorted(cases, key=lambda case: case.case_id)


class Evaluator:
    def __init__(self) -> None:
        self.repo = GraphRepository()
        self.resolver = KnowledgeResolver(self.repo)
        self.retriever = HybridRetriever()

    def evaluate(self, run_label: str = 'local') -> EvaluationReport:
        dimensions: list[DimensionResult] = []
        for case in load_eval_cases():
            dimensions.extend(self._score_case(case))
        # Same code, same data, same verdict: the label is derived from the results.
        digest = hashlib.sha256(
            json.dumps([d.model_dump(mode='json') for d in dimensions], sort_keys=True).encode()
        ).hexdigest()[:12]
        report = EvaluationReport(run_label=f'{run_label}:{digest}', dimensions=dimensions)
        self._persist(report)
        return report

    def _score_case(self, case: EvaluationCase) -> list[DimensionResult]:
        context = self.repo.load_application(case.application_id)
        knowledge = self.resolver.resolve(context)
        chunks = self.retriever.retrieve(case.question, context, knowledge) if knowledge.offered else []
        detail = {'case_id': case.case_id, 'application_id': case.application_id,
                  'chunk_ids': sorted(chunk.chunk_id for chunk in chunks)}

        missing = self.retriever.coverage(chunks, knowledge) if knowledge.offered else set()
        out_of_scope = sorted({chunk.chunk_id for chunk in chunks
                               if chunk.document_id not in knowledge.document_ids})
        conflicts = [conflict.as_dict() for conflict in knowledge.conflicts]

        return [
            DimensionResult(name='evidence_completeness',
                            passed=(not knowledge.offered) or not missing,
                            detail={**detail, 'missing_roles': sorted(missing),
                                    'required_roles': sorted(knowledge.required_roles)}),
            DimensionResult(name='scope_isolation',
                            passed=not out_of_scope,
                            detail={**detail, 'out_of_scope': out_of_scope}),
            DimensionResult(name='abstention_correctness',
                            passed=knowledge.offered or not chunks,
                            detail={**detail, 'offered': knowledge.offered,
                                    'offering_note': knowledge.offering_note}),
            DimensionResult(name='conflict_visibility',
                            passed=all(c.get('detail') for c in conflicts),
                            detail={**detail, 'conflicts': conflicts}),
            DimensionResult(name='citation_grounding',
                            passed=all(chunk.source_uri and chunk.revision_id for chunk in chunks),
                            detail={**detail, 'uncitable': sorted(chunk.chunk_id for chunk in chunks
                                                                  if not (chunk.source_uri and chunk.revision_id))}),
        ]

    def _persist(self, report: EvaluationReport) -> None:
        settings = get_settings()
        with get_conn() as conn:
            run = conn.execute(
                """
                INSERT INTO evaluation_runs(run_label, corpus_id, index_id, code_fingerprint, completed_at, result_payload)
                VALUES (%s, %s, %s, %s, now(), %s)
                RETURNING run_id
                """,
                (report.run_label, settings.corpus_id, settings.index_id, 'solution',
                 Json(report.model_dump(mode='json'))),
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


def summarise(report: EvaluationReport) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {name: {'passed': 0, 'failed': 0} for name in DIMENSIONS}
    for dim in report.dimensions:
        bucket = out.setdefault(dim.name, {'passed': 0, 'failed': 0})
        bucket['passed' if dim.passed else 'failed'] += 1
    return out


def main() -> None:
    report = Evaluator().evaluate('manual')
    print(json.dumps({'run_label': report.run_label, 'by_dimension': summarise(report)}, indent=2))


if __name__ == '__main__':
    main()
