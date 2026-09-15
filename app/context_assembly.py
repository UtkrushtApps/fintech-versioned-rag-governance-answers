from __future__ import annotations

import hashlib
from typing import Any

from app.graph_repository import GraphRepository
from app.models import Citation, EvidenceChunk, GraphAttribution, ProvenanceRecord


def fingerprint_text(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()[:16]


class ContextAssembler:
    def __init__(self) -> None:
        self.repo = GraphRepository()

    def assemble(
        self,
        request_id: str,
        application_id: str,
        chunks: list[EvidenceChunk],
        graph: GraphAttribution,
    ) -> tuple[str, list[Citation], ProvenanceRecord]:
        selected = chunks[:4]

        citations: list[Citation] = []
        parts: list[str] = []
        graph_node_ids = [node.node_id for node in graph.nodes]
        graph_edge_refs = [edge.get('edge_id', '') for edge in graph.edges]
        graph_refs = [ref for ref in (graph_node_ids + graph_edge_refs) if ref]

        for idx, chunk in enumerate(selected, start=1):
            parts.append(f'[{idx}] {chunk.text}')
            citations.append(
                Citation(
                    citation_id=f'C{idx}',
                    chunk_id=chunk.chunk_id,
                    revision_id=chunk.revision_id,
                    source_uri=chunk.source_uri,
                    passage=chunk.text[:320],
                    # Keep graph refs attributable but bounded.
                    graph_refs=graph_refs[: min(6, len(graph_refs))],
                )
            )

        context_text = '\n\n'.join(parts)

        revision_ids = sorted({chunk.revision_id for chunk in selected if chunk.revision_id})
        manifest_refs: list[str] = []
        index_id = selected[0].index_id if selected else None
        manifest_details: dict[str, Any] = {}
        upstream_prov_rows: list[dict[str, Any]] = []

        if index_id:
            manifest_refs = [index_id]
            manifest = self.repo.manifest(index_id)
            if manifest:
                manifest_details = {
                    'index_id': manifest.get('index_id'),
                    'collection_name': manifest.get('collection_name'),
                    'corpus_id': manifest.get('corpus_id'),
                    'embedding_model': manifest.get('embedding_model'),
                    'embedding_version': manifest.get('embedding_version'),
                    'built_at': str(manifest.get('built_at')),
                    'promoted_at': str(manifest.get('promoted_at')) if manifest.get('promoted_at') else None,
                    'lifecycle_state': manifest.get('lifecycle_state'),
                    'source_revision_count': manifest.get('source_revision_count'),
                    'manifest_hash': manifest.get('manifest_hash'),
                }

        if revision_ids:
            upstream_prov_rows = self.repo.provenance_for_revisions(revision_ids)

        provenance = ProvenanceRecord(
            request_id=request_id,
            application_id=application_id,
            index_id=index_id,
            chunk_ids=[chunk.chunk_id for chunk in selected],
            graph_refs=[ref for ref in graph_refs if ref],
            revision_ids=revision_ids,
            manifest_refs=manifest_refs,
            manifest_details=manifest_details,
            document_provenance_ids=[row['provenance_id'] for row in upstream_prov_rows if row.get('provenance_id')],
            upstream_provenance=upstream_prov_rows,
        )

        return context_text, citations, provenance
