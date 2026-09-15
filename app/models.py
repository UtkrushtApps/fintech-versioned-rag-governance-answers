from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class AnswerRequest(BaseModel):
    application_id: str
    question: str
    reviewer_id: str | None = None


class ApplicationContext(BaseModel):
    application_id: str
    product_id: str
    jurisdiction: str
    signed_at: datetime
    customer_id: str
    principal_amount: str


class EvidenceChunk(BaseModel):
    chunk_id: str
    text: str
    score: float = 0.0
    product_id: str | None = None
    jurisdiction: str | None = None
    document_id: str | None = None
    revision_id: str | None = None
    source_uri: str | None = None
    index_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphEntity(BaseModel):
    node_id: str
    node_type: str
    display_name: str
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphAttribution(BaseModel):
    nodes: list[GraphEntity] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)
    statements: list[dict[str, Any]] = Field(default_factory=list)


class Citation(BaseModel):
    citation_id: str
    chunk_id: str | None = None
    revision_id: str | None = None
    source_uri: str | None = None
    passage: str
    graph_refs: list[str] = Field(default_factory=list)


class ProvenanceRecord(BaseModel):
    request_id: str
    application_id: str
    index_id: str | None = None
    chunk_ids: list[str] = Field(default_factory=list)

    # Graph provenance
    graph_refs: list[str] = Field(default_factory=list)

    # Versioned regulatory provenance
    revision_ids: list[str] = Field(default_factory=list)

    # Index manifest provenance
    manifest_refs: list[str] = Field(default_factory=list)
    manifest_details: dict[str, Any] = Field(default_factory=dict)

    # Document-level upstream provenance for each selected revision
    document_provenance_ids: list[str] = Field(default_factory=list)
    upstream_provenance: list[dict[str, Any]] = Field(default_factory=list)


class AnswerResponse(BaseModel):
    request_id: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    provenance: ProvenanceRecord
    warnings: list[str] = Field(default_factory=list)


class EvaluationCase(BaseModel):
    case_id: str
    application_id: str
    question: str
    notes: str | None = None


class DimensionResult(BaseModel):
    name: str
    passed: bool
    detail: dict[str, Any] = Field(default_factory=dict)


class EvaluationReport(BaseModel):
    run_label: str
    dimensions: list[DimensionResult]
