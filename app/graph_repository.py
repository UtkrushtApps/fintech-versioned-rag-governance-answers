from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator

import psycopg
from psycopg.rows import dict_row

from app.config import get_settings
from app.models import ApplicationContext


@contextmanager
def get_conn() -> Iterator[psycopg.Connection]:
    conn = psycopg.connect(get_settings().database_url, row_factory=dict_row)
    try:
        yield conn
    finally:
        conn.close()


class GraphRepository:
    def load_application(self, application_id: str) -> ApplicationContext:
        with get_conn() as conn:
            row = conn.execute(
                """
                SELECT application_id, product_id, jurisdiction, signed_at,
                       customer_id, principal_amount::text AS principal_amount
                FROM loan_applications
                WHERE application_id = %s
                """,
                (application_id,),
            ).fetchone()
        if not row:
            raise ValueError(f'Unknown application_id: {application_id}')
        return ApplicationContext(**row)

    def product_node(self, product_id: str, signed_at: datetime | None = None) -> dict[str, Any] | None:
        # Optional time scoping prevents leakage across product lifecycle validity windows.
        with get_conn() as conn:
            if signed_at is None:
                return conn.execute(
                    """
                    SELECT n.*
                    FROM entity_crosswalks x
                    JOIN graph_nodes n ON n.node_id = x.node_id
                    WHERE x.source_system = 'postgres.products' AND x.source_key = %s
                    ORDER BY x.confidence DESC, n.node_id
                    LIMIT 1
                    """,
                    (product_id,),
                ).fetchone()

            return conn.execute(
                """
                SELECT n.*
                FROM entity_crosswalks x
                JOIN graph_nodes n ON n.node_id = x.node_id
                WHERE x.source_system = 'postgres.products'
                  AND x.source_key = %s
                  AND (x.valid_from IS NULL OR x.valid_from <= %s)
                  AND (x.valid_to IS NULL OR x.valid_to > %s)
                  AND (n.valid_from IS NULL OR n.valid_from <= %s)
                  AND (n.valid_to IS NULL OR n.valid_to > %s)
                ORDER BY x.confidence DESC, n.node_id
                LIMIT 1
                """,
                (product_id, signed_at, signed_at, signed_at, signed_at),
            ).fetchone()

    def revision_node(self, revision_id: str, signed_at: datetime | None = None) -> dict[str, Any] | None:
        # Revision nodes are typically unique; time scoping avoids stray archived periods.
        with get_conn() as conn:
            if signed_at is None:
                return conn.execute(
                    """
                    SELECT n.*
                    FROM entity_crosswalks x
                    JOIN graph_nodes n ON n.node_id = x.node_id
                    WHERE x.source_system = 'postgres.document_revisions' AND x.source_key = %s
                    ORDER BY x.confidence DESC, n.node_id
                    LIMIT 1
                    """,
                    (revision_id,),
                ).fetchone()

            return conn.execute(
                """
                SELECT n.*
                FROM entity_crosswalks x
                JOIN graph_nodes n ON n.node_id = x.node_id
                WHERE x.source_system = 'postgres.document_revisions'
                  AND x.source_key = %s
                  AND (x.valid_from IS NULL OR x.valid_from <= %s)
                  AND (x.valid_to IS NULL OR x.valid_to > %s)
                  AND (n.valid_from IS NULL OR n.valid_from <= %s)
                  AND (n.valid_to IS NULL OR n.valid_to > %s)
                ORDER BY x.confidence DESC, n.node_id
                LIMIT 1
                """,
                (revision_id, signed_at, signed_at, signed_at, signed_at),
            ).fetchone()

    def disclosure_revisions_for_context(
        self,
        *,
        product_id: str,
        jurisdiction: str,
        signed_at: datetime,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        # Versioned disclosure revisions authorized at signing.
        with get_conn() as conn:
            return list(
                conn.execute(
                    """
                    SELECT r.revision_id, r.document_id, r.revision_label, r.effective_at,
                           r.superseded_at, r.lifecycle_state, r.source_uri
                    FROM document_revisions r
                    JOIN source_documents d ON d.document_id = r.document_id
                    WHERE d.product_id = %s
                      AND d.jurisdiction = %s
                      AND r.effective_at <= %s
                      AND (r.superseded_at IS NULL OR r.superseded_at > %s)
                    ORDER BY r.effective_at DESC, r.published_at DESC, r.revision_id
                    LIMIT %s
                    """,
                    (product_id, jurisdiction, signed_at, signed_at, limit),
                )
            )

    def disclosure_revision_for_context(
        self,
        *,
        product_id: str,
        jurisdiction: str,
        signed_at: datetime,
    ) -> dict[str, Any] | None:
        revisions = self.disclosure_revisions_for_context(
            product_id=product_id,
            jurisdiction=jurisdiction,
            signed_at=signed_at,
            limit=1,
        )
        return revisions[0] if revisions else None

    def disclosure_document_ids_for_context(
        self,
        *,
        product_id: str,
        jurisdiction: str,
        signed_at: datetime,
    ) -> list[str]:
        with get_conn() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT d.document_id
                FROM source_documents d
                JOIN document_revisions r ON r.document_id = d.document_id
                WHERE d.product_id = %s
                  AND d.jurisdiction = %s
                  AND r.effective_at <= %s
                  AND (r.superseded_at IS NULL OR r.superseded_at > %s)
                """,
                (product_id, jurisdiction, signed_at, signed_at),
            ).fetchall()
        return [row['document_id'] for row in rows]

    def document_node_from_revision_node(self, revision_node_id: str) -> dict[str, Any] | None:
        # Graph edge HAS_REVISION links DisclosureDocument -> DisclosureRevision.
        with get_conn() as conn:
            row = conn.execute(
                """
                SELECT n.*
                FROM graph_edges e
                JOIN graph_nodes n ON n.node_id = e.from_node_id
                WHERE e.relationship_type = 'HAS_REVISION'
                  AND e.to_node_id = %s
                ORDER BY e.valid_from DESC NULLS LAST, e.edge_id
                LIMIT 1
                """,
                (revision_node_id,),
            ).fetchone()
        # Prefer DisclosureDocument nodes.
        if row and row.get('node_type') != 'DisclosureDocument':
            return None
        return row

    def edges_for_nodes(self, node_ids: list[str]) -> list[dict[str, Any]]:
        if not node_ids:
            return []
        with get_conn() as conn:
            return list(
                conn.execute(
                    """
                    SELECT edge_id, from_node_id, to_node_id, relationship_type,
                           valid_from, valid_to, assertion_status, confidence, provenance_ref
                    FROM graph_edges
                    WHERE from_node_id = ANY(%s) OR to_node_id = ANY(%s)
                    ORDER BY edge_id
                    """,
                    (node_ids, node_ids),
                )
            )

    def manifest(self, index_id: str) -> dict[str, Any] | None:
        with get_conn() as conn:
            return conn.execute(
                "SELECT * FROM index_manifests WHERE index_id = %s",
                (index_id,),
            ).fetchone()

    def provenance_for_revisions(self, revision_ids: list[str]) -> list[dict[str, Any]]:
        if not revision_ids:
            return []
        with get_conn() as conn:
            return list(
                conn.execute(
                    """
                    SELECT dp.provenance_id, dp.revision_id, dp.upstream_system, dp.upstream_record_id,
                           dp.collected_at, dp.lineage_hash, dp.notes
                    FROM document_provenance dp
                    WHERE dp.revision_id = ANY(%s)
                    ORDER BY dp.revision_id, dp.collected_at
                    """,
                    (revision_ids,),
                )
            )

    # Backwards-compatibility for older code paths (not used in hardened versioned retrieval).
    def latest_disclosure_revision(self, product_id: str) -> dict[str, Any] | None:
        with get_conn() as conn:
            return conn.execute(
                """
                SELECT r.revision_id, r.document_id, r.revision_label, r.effective_at,
                       r.superseded_at, r.lifecycle_state, r.source_uri,
                       d.jurisdiction, d.product_id
                FROM document_revisions r
                JOIN source_documents d ON d.document_id = r.document_id
                WHERE d.product_id = %s
                ORDER BY r.effective_at DESC, r.published_at DESC, r.revision_id
                LIMIT 1
                """,
                (product_id,),
            ).fetchone()
