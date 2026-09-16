from __future__ import annotations

from contextlib import contextmanager
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

    def product_nodes(self, product_id: str) -> list[dict[str, Any]]:
        """Every product node the key maps to. More than one means the records disagree."""
        with get_conn() as conn:
            return list(
                conn.execute(
                    """
                    SELECT n.*, x.confidence
                    FROM entity_crosswalks x
                    JOIN graph_nodes n ON n.node_id = x.node_id
                    WHERE x.source_system = 'postgres.products' AND x.source_key = %s
                    ORDER BY x.confidence DESC, n.node_id
                    """,
                    (product_id,),
                )
            )

    def product_node(self, product_id: str) -> dict[str, Any] | None:
        with get_conn() as conn:
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

    def revision_node(self, revision_id: str) -> dict[str, Any] | None:
        with get_conn() as conn:
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

    def offerings_for(self, product_id: str) -> list[dict[str, Any]]:
        with get_conn() as conn:
            return list(
                conn.execute(
                    """
                    SELECT offering_id, product_id, jurisdiction, offered_from, offered_to, offering_status
                    FROM product_offerings
                    WHERE product_id = %s
                    ORDER BY jurisdiction, offered_from
                    """,
                    (product_id,),
                )
            )

    def override_rules_for(self, document_id: str) -> list[dict[str, Any]]:
        with get_conn() as conn:
            return list(
                conn.execute(
                    """
                    SELECT rule_id, base_document_id, addendum_document_id, jurisdiction, overrides_topics
                    FROM override_rules
                    WHERE base_document_id = %s
                    ORDER BY jurisdiction
                    """,
                    (document_id,),
                )
            )

    def documents_for(self, product_id: str) -> list[dict[str, Any]]:
        with get_conn() as conn:
            return list(
                conn.execute(
                    """
                    SELECT document_id, document_type, title, jurisdiction, lifecycle_state
                    FROM source_documents
                    WHERE product_id = %s
                    ORDER BY document_type, jurisdiction
                    """,
                    (product_id,),
                )
            )

    def history_for(self, product_id: str) -> list[dict[str, Any]]:
        with get_conn() as conn:
            return list(
                conn.execute(
                    """
                    SELECT history_id, predecessor_product_id, successor_product_id, change_type, changed_on, note
                    FROM product_history
                    WHERE predecessor_product_id = %s OR successor_product_id = %s
                    ORDER BY changed_on
                    """,
                    (product_id, product_id),
                )
            )

    def governance_for(self, subject_id: str) -> list[dict[str, Any]]:
        with get_conn() as conn:
            return list(
                conn.execute(
                    """
                    SELECT governance_id, subject_type, subject_id, state_label, reason, starts_at, ends_at
                    FROM governance_state
                    WHERE subject_id = %s
                    ORDER BY starts_at
                    """,
                    (subject_id,),
                )
            )

    def manifest(self, index_id: str) -> dict[str, Any] | None:
        with get_conn() as conn:
            return conn.execute(
                "SELECT * FROM index_manifests WHERE index_id = %s",
                (index_id,),
            ).fetchone()
