from __future__ import annotations

import hashlib

from app.models import Citation, EvidenceChunk, GraphAttribution, ProvenanceRecord


def fingerprint_text(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()[:16]


class ContextAssembler:
    def assemble(self, request_id: str, application_id: str, chunks: list[EvidenceChunk], graph: GraphAttribution) -> tuple[str, list[Citation], ProvenanceRecord]:
        selected = chunks[:4]
        parts: list[str] = []
        citations: list[Citation] = []
        graph_refs = [node.node_id for node in graph.nodes] + [edge.get('edge_id', '') for edge in graph.edges]
        for idx, chunk in enumerate(selected, start=1):
            parts.append(f'[{idx}] {chunk.text}')
            citations.append(
                Citation(
                    citation_id=f'C{idx}',
                    chunk_id=chunk.chunk_id,
                    revision_id=chunk.revision_id,
                    source_uri=chunk.source_uri,
                    passage=chunk.text[:320],
                    graph_refs=[ref for ref in graph_refs if ref],
                )
            )
        context_text = '\n\n'.join(parts)
        provenance = ProvenanceRecord(
            request_id=request_id,
            application_id=application_id,
            index_id=selected[0].index_id if selected else None,
            chunk_ids=[chunk.chunk_id for chunk in selected],
            graph_refs=[ref for ref in graph_refs if ref],
            revision_ids=sorted({chunk.revision_id for chunk in selected if chunk.revision_id}),
            manifest_refs=[selected[0].index_id] if selected and selected[0].index_id else [],
        )
        return context_text, citations, provenance
