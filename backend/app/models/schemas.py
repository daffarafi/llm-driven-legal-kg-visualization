"""Pydantic models for API request/response schemas."""

from pydantic import BaseModel, Field


# --- Graph ---

class GraphNode(BaseModel):
    id: str
    labels: list[str]
    label: str | None = None
    content: str | None = None


class GraphEdge(BaseModel):
    source: str
    target: str
    type: str


class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


# --- Node Detail ---

class NodeRelation(BaseModel):
    type: str
    direction: str
    target_id: str | None = None
    target_label: str | None = None
    target_type: list[str] | None = None
    source_id: str | None = None
    source_label: str | None = None
    source_type: list[str] | None = None


class NodeDetailResponse(BaseModel):
    id: str
    labels: list[str]
    properties: dict
    outgoing: list[NodeRelation]
    incoming: list[NodeRelation]


# --- Search ---

class SearchResult(BaseModel):
    id: str
    labels: list[str]
    label: str | None = None
    content: str | None = None


class SearchResponse(BaseModel):
    results: list[SearchResult]
    total: int


# --- QA ---

class QARequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    doc_ids: list[str] | None = Field(None, description="Filter KG context to these source_document_ids. None = all documents.")


class QAProcessStep(BaseModel):
    step: int
    label: str
    detail: str
    status: str = "done"


class QAGraphNode(BaseModel):
    id: str
    labels: list[str]
    label: str | None = None


class QAGraphEdge(BaseModel):
    source: str
    target: str
    type: str


class QAResponse(BaseModel):
    answer: str
    cypher_query: str
    kg_context: list[dict]
    references: list[str]
    process_steps: list[QAProcessStep]
    graph: dict = Field(default_factory=lambda: {"nodes": [], "edges": []})


# --- Stats ---

class TypeCount(BaseModel):
    label: str
    count: int


class StatsResponse(BaseModel):
    total_nodes: int
    total_edges: int
    node_types: list[TypeCount]
    edge_types: list[TypeCount]


# --- Document ---

class DocumentSection(BaseModel):
    id: str | None = None
    label: str | None = None
    content: str | None = None
    bab: str | None = None
    pasal: str | None = None


class DocumentResponse(BaseModel):
    document: dict
    bab: list[DocumentSection]
    pasal: list[DocumentSection]
    ayat: list[DocumentSection]
