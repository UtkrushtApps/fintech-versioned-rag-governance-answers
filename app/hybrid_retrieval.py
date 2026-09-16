from __future__ import annotations

from app.dense_retrieval import DenseRetriever
from app.knowledge import ApplicableKnowledge
from app.models import ApplicationContext, EvidenceChunk
from app.sparse_retrieval import SparseRetriever

PER_ROLE = 2
TOTAL = 6


class HybridRetriever:
    def __init__(self) -> None:
        self.dense = DenseRetriever()
        self.sparse = SparseRetriever()

    def _merge(self, question: str, context: ApplicationContext) -> list[EvidenceChunk]:
        by_id: dict[str, EvidenceChunk] = {}
        for rank, chunk in enumerate(self.dense.search(question, context, limit=24)):
            chunk.score += max(0.0, 1.0 - rank * 0.02)
            by_id[chunk.chunk_id] = chunk
        for rank, chunk in enumerate(self.sparse.search(question, context, limit=24)):
            if chunk.chunk_id in by_id:
                by_id[chunk.chunk_id].score += max(0.0, 0.7 - rank * 0.02)
            else:
                chunk.score += max(0.0, 0.7 - rank * 0.02)
                by_id[chunk.chunk_id] = chunk
        return sorted(by_id.values(), key=lambda item: (-item.score, item.chunk_id))

    def retrieve(self, question: str, context: ApplicationContext,
                 knowledge: ApplicableKnowledge | None = None) -> list[EvidenceChunk]:
        ranked = self._merge(question, context)
        if knowledge is None:
            return ranked[:TOTAL]

        # Only evidence from the documents the bank's records say apply here.
        eligible = [chunk for chunk in ranked if chunk.document_id in knowledge.document_ids]

        # Every required role gets room in the context, so an answer that needs the
        # national schedule and the borrower's state addendum can have both.
        selected: list[EvidenceChunk] = []
        for role in sorted(knowledge.required_roles):
            for chunk in eligible:
                if chunk.document_role == role and chunk not in selected:
                    selected.append(chunk)
                    if sum(1 for c in selected if c.document_role == role) >= PER_ROLE:
                        break
        for chunk in eligible:
            if len(selected) >= TOTAL:
                break
            if chunk not in selected:
                selected.append(chunk)
        return sorted(selected, key=lambda item: (-item.score, item.chunk_id))[:TOTAL]

    def coverage(self, chunks: list[EvidenceChunk], knowledge: ApplicableKnowledge) -> set[str]:
        """Required roles with no evidence in the context — an incomplete answer."""
        present = {chunk.document_role for chunk in chunks}
        return {role for role in knowledge.required_roles if role not in present}
