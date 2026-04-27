/* TypeScript interfaces for the Legal KG Visualization app */

// --- Graph ---

export interface GraphNode {
  id: string;
  labels: string[];
  label?: string;
  content?: string;
  node_type?: string;
  source_document_id?: string;
}

export interface GraphEdge {
  source: string;
  target: string;
  type: string;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

// --- Node Detail ---

export interface NodeRelation {
  type: string;
  direction: string;
  target_id?: string;
  target_label?: string;
  target_type?: string[];
  source_id?: string;
  source_label?: string;
  source_type?: string[];
}

export interface NodeDetail {
  id: string;
  labels: string[];
  properties: Record<string, unknown>;
  outgoing: NodeRelation[];
  incoming: NodeRelation[];
}

// --- Search ---

export interface SearchResult {
  id: string;
  labels: string[];
  label?: string;
  content?: string;
}

// --- QA ---

export interface QAProcessStep {
  step: number;
  label: string;
  detail: string;
  status: string;
}

export interface QAResponse {
  answer: string;
  cypher_query: string;
  kg_context: Record<string, unknown>[];
  references: string[];
  process_steps: QAProcessStep[];
  graph?: {
    nodes: { id: string; labels: string[]; label?: string }[];
    edges: { source: string; target: string; type: string }[];
  };
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  qa_response?: QAResponse;
  timestamp: Date;
}

// --- Stats ---

export interface TypeCount {
  label: string;
  count: number;
}

export interface StatsData {
  total_nodes: number;
  total_edges: number;
  node_types: TypeCount[];
  edge_types: TypeCount[];
}

// --- Document ---

export interface DocumentSection {
  id?: string;
  label?: string;
  content?: string;
  bab?: string;
  pasal?: string;
}

export interface DocumentData {
  document: Record<string, unknown>;
  bab?: DocumentSection[];
  pasal?: DocumentSection[];
  ayat?: DocumentSection[];
  entities_by_type?: Record<string, DocumentSection[]>;
  total_entities?: number;
}

// --- Regulations (Multi-document) ---

export interface Regulation {
  doc_id: string;
  label: string;
  short_name?: string;
  regulation_type?: string;
  number?: string;
  year?: number;
  status?: string;
  entity_count?: number;
  source_document_id?: string;
}

export interface RegulationEdge {
  source: string;
  target: string;
  type: string;
  description?: string;
}

export interface RegulationGraph {
  nodes: Regulation[];
  edges: RegulationEdge[];
}

export interface Amendment {
  id: string;
  label: string;
  version?: number;
  status?: string;
  source_doc?: string;
  amended_to_id?: string;
  amended_to_label?: string;
}

// --- Graph Viz ---

export interface ForceGraphNode extends GraphNode {
  x?: number;
  y?: number;
  color?: string;
  val?: number;
}

// Node type color mapping
export const NODE_COLORS: Record<string, string> = {
  Regulasi: "#3b82f6",       // blue
  Bab: "#8b5cf6",            // purple
  Bagian: "#a78bfa",         // light purple
  Pasal: "#22c55e",          // green
  Ayat: "#86efac",           // light green
  EntitasHukum: "#f97316",   // orange
  PerbuatanHukum: "#ef4444", // red
  Sanksi: "#dc2626",         // dark red
  KonsepHukum: "#eab308",    // yellow
  VersiPasal: "#14b8a6",     // teal
};

export const NODE_SIZES: Record<string, number> = {
  Regulasi: 8,
  Bab: 5,
  Bagian: 4,
  Pasal: 4,
  Ayat: 3,
  EntitasHukum: 5,
  PerbuatanHukum: 4,
  Sanksi: 4,
  KonsepHukum: 4,
  VersiPasal: 5,
};

// Document source color mapping for multi-doc visualization
export const DOC_COLORS: Record<string, string> = {
  UU_11_2008: "#3b82f6",
  UU_19_2016: "#6366f1",
  UU_27_2022: "#8b5cf6",
  UU_36_1999: "#a855f7",
  PP_71_2019: "#22c55e",
  PP_80_2019: "#16a34a",
  PP_82_2012: "#15803d",
  Perpres_95_2018: "#f97316",
  Perpres_132_2022: "#ea580c",
  Permen_Kominfo_5_2017: "#eab308",
  POJK_11_2022: "#06b6d4",
};
