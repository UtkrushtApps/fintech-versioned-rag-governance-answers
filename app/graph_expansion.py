from __future__ import annotations

from app.graph_repository import GraphRepository
from app.knowledge import ApplicableKnowledge
from app.models import ApplicationContext, EvidenceChunk, GraphAttribution, GraphEntity


class GraphExpander:
    def __init__(self) -> None:
        self.repo = GraphRepository()

    def expand(self, context: ApplicationContext, chunks: list[EvidenceChunk],
               knowledge: ApplicableKnowledge | None = None) -> GraphAttribution:
        nodes: list[GraphEntity] = []
        node_ids: list[str] = []
        statements: list[dict] = []

        product_rows = knowledge.product_nodes if knowledge else self.repo.product_nodes(context.product_id)
        for row in product_rows:
            node_ids.append(row['node_id'])
            nodes.append(GraphEntity(node_id=row['node_id'], node_type=row['node_type'],
                                     display_name=row['display_name'], properties=row.get('properties') or {}))

        # Attribution names the revisions the answer was actually built from.
        used = {chunk.revision_id for chunk in chunks if chunk.revision_id}
        for revision_id in sorted(used):
            row = self.repo.revision_node(revision_id)
            if not row:
                continue
            node_ids.append(row['node_id'])
            nodes.append(GraphEntity(node_id=row['node_id'], node_type=row['node_type'],
                                     display_name=row['display_name'], properties=row.get('properties') or {}))
        if knowledge:
            for document in knowledge.documents:
                statements.append({'document_id': document['document_id'], 'role': document['role'],
                                   'jurisdiction': document['jurisdiction'],
                                   'revision_id': document.get('revision_id'),
                                   'source_uri': document.get('source_uri')})
            for conflict in knowledge.conflicts:
                statements.append({'conflict': conflict.kind, 'detail': conflict.detail,
                                   'sources': conflict.sources})
            for change in knowledge.history:
                statements.append({'history': change['change_type'],
                                   'from': change['predecessor_product_id'],
                                   'to': change['successor_product_id'],
                                   'changed_on': str(change['changed_on'])})
        edges = self.repo.edges_for_nodes(node_ids)
        return GraphAttribution(nodes=nodes, edges=edges, statements=statements)
