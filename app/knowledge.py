"""What the bank's own records say applies to an application.

The disclosure text alone cannot answer a reviewer's question: which product an
application really is, whether the bank ever offered it in that state, and which
documents have to be read together are all facts held in the catalogue and the
knowledge graph. This module resolves them once per request and reports the
disagreements it finds instead of choosing between them.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from app.graph_repository import GraphRepository, get_conn
from app.models import ApplicationContext

NATIONAL = 'ALL'
BASE_ROLES = ('fee_schedule', 'rate_sheet', 'servicing_terms')


@dataclass
class Conflict:
    kind: str
    detail: str
    sources: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {'kind': self.kind, 'detail': self.detail, 'sources': self.sources}


@dataclass
class ApplicableKnowledge:
    context: ApplicationContext
    offered: bool
    offering_note: str
    documents: list[dict[str, Any]]
    required_roles: set[str]
    conflicts: list[Conflict]
    product_nodes: list[dict[str, Any]]
    history: list[dict[str, Any]]

    @property
    def document_ids(self) -> set[str]:
        return {doc['document_id'] for doc in self.documents}

    @property
    def revision_ids(self) -> set[str]:
        return {doc['revision_id'] for doc in self.documents if doc.get('revision_id')}

    def fingerprint(self) -> str:
        """Identity of the knowledge this answer rests on.

        Any correction to the catalogue, the documents or the graph changes this
        value, so anything keyed on it stops being reused the moment the records
        change.
        """
        payload = {
            'application_id': self.context.application_id,
            'offered': self.offered,
            'documents': sorted((d['document_id'], d.get('revision_id')) for d in self.documents),
            'roles': sorted(self.required_roles),
            'conflicts': sorted(c.kind + '|' + c.detail for c in self.conflicts),
            'product_nodes': sorted(n['node_id'] for n in self.product_nodes),
            'history': sorted(h['history_id'] for h in self.history),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


class KnowledgeResolver:
    def __init__(self, repo: GraphRepository | None = None) -> None:
        self.repo = repo or GraphRepository()

    # -- offering ---------------------------------------------------------
    def _offering(self, context: ApplicationContext, conflicts: list[Conflict]) -> tuple[bool, str]:
        rows = [row for row in self.repo.offerings_for(context.product_id)
                if row['jurisdiction'] == context.jurisdiction]
        governance = [row for row in self.repo.governance_for(f'{context.product_id}:{context.jurisdiction}')
                      if row['ends_at'] is None]
        live = [row for row in rows
                if row['offering_status'] == 'offered'
                and row['offered_from'] <= context.signed_at
                and (row['offered_to'] is None or context.signed_at < row['offered_to'])]

        if rows and any(row['offering_status'] != 'offered' for row in rows):
            conflicts.append(Conflict(
                'offering_status',
                f'The catalogue records {context.product_id} in {context.jurisdiction} as '
                + ', '.join(sorted({row['offering_status'] for row in rows})),
                ['postgres.product_offerings'],
            ))
        if governance:
            labels = ', '.join(sorted({row['state_label'] for row in governance}))
            conflicts.append(Conflict(
                'governance_vs_catalogue',
                f'The compliance register records {context.product_id} in {context.jurisdiction} as {labels}',
                ['postgres.governance_state'] + (['postgres.product_offerings'] if rows else []),
            ))
            return False, f'The compliance register records this product in {context.jurisdiction} as {labels}.'
        if live:
            return True, f'Offered in {context.jurisdiction} since {live[0]["offered_from"]:%Y-%m-%d}.'
        if rows:
            return False, (f'The only offerings recorded for {context.product_id} in {context.jurisdiction} '
                           f'do not cover the signing date or are not in force.')
        offered_in = sorted({row['jurisdiction'] for row in self.repo.offerings_for(context.product_id)
                             if row['offering_status'] == 'offered'})
        return False, (f'No offering of {context.product_id} in {context.jurisdiction} is recorded; '
                       f'the catalogue lists {", ".join(offered_in) or "no jurisdictions"}.')

    # -- documents --------------------------------------------------------
    def _documents(self, context: ApplicationContext, conflicts: list[Conflict]) -> tuple[list[dict[str, Any]], set[str]]:
        with get_conn() as conn:
            rows = list(conn.execute(
                """
                SELECT d.document_id, d.document_type AS role, d.jurisdiction, d.title,
                       r.revision_id, r.revision_label, r.effective_at, r.superseded_at,
                       r.lifecycle_state, r.source_uri
                FROM source_documents d
                JOIN document_revisions r ON r.document_id = d.document_id
                WHERE d.product_id = %s
                  AND d.jurisdiction IN (%s, %s)
                  AND r.effective_at <= %s
                  AND (r.superseded_at IS NULL OR r.superseded_at > %s)
                ORDER BY d.document_type, d.jurisdiction, r.effective_at DESC
                """,
                (context.product_id, NATIONAL, context.jurisdiction, context.signed_at, context.signed_at),
            ))
        documents: dict[str, dict[str, Any]] = {}
        for row in rows:
            documents.setdefault(row['document_id'], row)

        # Documents that have to be read TOGETHER: a national document the borrower's
        # state overrides, and the addendum that overrides it. Other applicable
        # documents may still be used; these two are what an answer cannot omit.
        required: set[str] = set()
        for base in [d for d in documents.values() if d['role'] in BASE_ROLES]:
            for rule in self.repo.override_rules_for(base['document_id']):
                if rule['jurisdiction'] != context.jurisdiction:
                    continue
                required.add(base['role'])
                required.add('state_addendum')
                if rule['addendum_document_id'] not in documents:
                    conflicts.append(Conflict(
                        'missing_addendum',
                        f'{base["document_id"]} is overridden in {context.jurisdiction} by '
                        f'{rule["addendum_document_id"]}, which has no revision in force at signing',
                        ['postgres.override_rules'],
                    ))
        if not required:
            required = {role for role in ('fee_schedule',) if any(d['role'] == role for d in documents.values())}
        return list(documents.values()), required

    # -- disagreements in the graph ---------------------------------------
    def _graph(self, context: ApplicationContext, documents: list[dict[str, Any]],
               conflicts: list[Conflict]) -> list[dict[str, Any]]:
        nodes = self.repo.product_nodes(context.product_id)
        if len(nodes) > 1:
            conflicts.append(Conflict(
                'product_identity',
                'The product key maps to more than one product in the knowledge graph: '
                + ', '.join(f'{n["display_name"]} ({n["node_id"]})' for n in nodes),
                ['postgres.entity_crosswalks'],
            ))
        node_ids = [n['node_id'] for n in nodes]
        for edge in self.repo.edges_for_nodes(node_ids):
            if edge['assertion_status'] != 'asserted':
                conflicts.append(Conflict(
                    'disputed_relationship',
                    f'{edge["relationship_type"]} {edge["from_node_id"]} -> {edge["to_node_id"]} '
                    f'is recorded as {edge["assertion_status"]}',
                    ['postgres.graph_edges'],
                ))
        # An override that still points at a revision no longer in force is a disagreement
        # between the compliance register and the document record.
        live_revisions = {doc['revision_id'] for doc in documents}
        with get_conn() as conn:
            stale = list(conn.execute(
                """
                SELECT e.edge_id, e.from_node_id, n.canonical_key AS revision_id
                FROM graph_edges e
                JOIN graph_nodes n ON n.node_id = e.from_node_id AND n.node_type = 'DisclosureRevision'
                JOIN document_revisions r ON r.revision_id = n.canonical_key
                JOIN source_documents d ON d.document_id = r.document_id
                WHERE e.relationship_type = 'OVERRIDDEN_BY'
                  AND d.product_id = %s
                  AND (e.properties ->> 'jurisdiction') = %s
                """,
                (context.product_id, context.jurisdiction),
            ))
        for row in stale:
            if row['revision_id'] not in live_revisions:
                conflicts.append(Conflict(
                    'stale_override_link',
                    f'{row["edge_id"]} links the {context.jurisdiction} addendum to {row["revision_id"]}, '
                    'which is not the revision in force at signing',
                    ['postgres.graph_edges', 'postgres.document_revisions'],
                ))
        return nodes

    def resolve(self, context: ApplicationContext) -> ApplicableKnowledge:
        conflicts: list[Conflict] = []
        offered, note = self._offering(context, conflicts)
        documents, required = self._documents(context, conflicts)
        nodes = self._graph(context, documents, conflicts)
        history = self.repo.history_for(context.product_id)
        return ApplicableKnowledge(
            context=context,
            offered=offered,
            offering_note=note,
            documents=documents,
            required_roles=required,
            conflicts=conflicts,
            product_nodes=nodes,
            history=history,
        )
