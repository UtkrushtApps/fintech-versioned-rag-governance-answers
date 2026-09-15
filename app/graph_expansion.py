from __future__ import annotations

from app.graph_repository import GraphRepository
from app.models import ApplicationContext, EvidenceChunk, GraphAttribution, GraphEntity


class GraphExpander:
    def __init__(self) -> None:
        self.repo = GraphRepository()

    def expand(self, context: ApplicationContext, chunks: list[EvidenceChunk]) -> GraphAttribution:
        # Hardened expansion: choose disclosure graph revision scoped to the application signing time.
        product = self.repo.product_node(context.product_id, signed_at=context.signed_at)
        revision_row = self.repo.disclosure_revision_for_context(
            product_id=context.product_id,
            jurisdiction=context.jurisdiction,
            signed_at=context.signed_at,
        )

        revision_node = self.repo.revision_node(revision_row['revision_id'], signed_at=context.signed_at) if revision_row else None
        doc_node = (
            self.repo.document_node_from_revision_node(revision_node['node_id'])
            if revision_node
            else None
        )

        nodes: list[GraphEntity] = []
        node_ids: list[str] = []

        for row in (product, doc_node, revision_node):
            if row:
                node_ids.append(row['node_id'])
                nodes.append(
                    GraphEntity(
                        node_id=row['node_id'],
                        node_type=row['node_type'],
                        display_name=row['display_name'],
                        properties=row.get('properties') or {},
                    )
                )

        edges = self.repo.edges_for_nodes(node_ids)

        statements: list[dict] = []
        if revision_row:
            statements.append(
                {
                    'revision_id': revision_row['revision_id'],
                    'revision_label': revision_row.get('revision_label'),
                    'source_uri': revision_row.get('source_uri'),
                    'effective_at': str(revision_row.get('effective_at')),
                    'superseded_at': str(revision_row.get('superseded_at')) if revision_row.get('superseded_at') else None,
                    'jurisdiction': context.jurisdiction,
                }
            )

        # Also record that the answer was scoped via the authorized signing context.
        statements.append(
            {
                'scope': 'application_signing_context',
                'application_id': context.application_id,
                'product_id': context.product_id,
                'jurisdiction': context.jurisdiction,
                'signed_at': context.signed_at.isoformat(),
            }
        )

        return GraphAttribution(nodes=nodes, edges=edges, statements=statements)
