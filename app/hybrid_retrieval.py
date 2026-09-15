from __future__ import annotations

from typing import Iterable

from app.dense_retrieval import DenseRetriever
from app.graph_repository import GraphRepository
from app.models import ApplicationContext, EvidenceChunk
from app.sparse_retrieval import SparseRetriever


class HybridRetriever:
    def __init__(self) -> None:
        self.dense = DenseRetriever()
        self.sparse = SparseRetriever()
        self.repo = GraphRepository()

    def _filter_by_authorized_versioned_scope(
        self,
        chunks: Iterable[EvidenceChunk],
        context: ApplicationContext,
    ) -> list[EvidenceChunk]:
        authorized_revisions = {
            r['revision_id']
            for r in self.repo.disclosure_revisions_for_context(
                product_id=context.product_id,
                jurisdiction=context.jurisdiction,
                signed_at=context.signed_at,
                limit=5,
            )
            if r.get('revision_id')
        }
        authorized_documents = set(
            self.repo.disclosure_document_ids_for_context(
                product_id=context.product_id,
                jurisdiction=context.jurisdiction,
                signed_at=context.signed_at,
            )
        )

        def keep(c: EvidenceChunk) -> bool:
            if c.revision_id:
                return c.revision_id in authorized_revisions
            # Fallback only if we can map by document_id.
            if c.document_id:
                return c.document_id in authorized_documents
            # If we cannot version-map the evidence, reject rather than risk leakage.
            return False

        filtered = [c for c in chunks if keep(c)]

        # Defensive fallback: if nothing survived strict version-mapping,
        # allow only exact product/jurisdiction matches that at least reduce leakage.
        if not filtered:
            fallback = [
                c
                for c in chunks
                if (c.product_id == context.product_id or c.product_id is None)
                and (c.jurisdiction == context.jurisdiction or c.jurisdiction is None)
            ]
            return fallback[:6]

        return filtered[:6]

    def retrieve(self, question: str, context: ApplicationContext) -> list[EvidenceChunk]:
        dense_hits = self.dense.search(question, context)
        sparse_hits = self.sparse.search(question, context)

        # Merge deterministically by chunk_id.
        by_id: dict[str, EvidenceChunk] = {}
        for rank, chunk in enumerate(dense_hits):
            chunk.score += max(0.0, 1.0 - rank * 0.05)
            by_id[chunk.chunk_id] = chunk

        for rank, chunk in enumerate(sparse_hits):
            if chunk.chunk_id in by_id:
                by_id[chunk.chunk_id].score += max(0.0, 0.7 - rank * 0.04)
            else:
                chunk.score += max(0.0, 0.7 - rank * 0.04)
                by_id[chunk.chunk_id] = chunk

        ranked = sorted(by_id.values(), key=lambda item: (-item.score, item.chunk_id))

        # Harden versioned disclosure: only evidence that is authorized at signing.
        scoped = self._filter_by_authorized_versioned_scope(ranked, context)
        return scoped
