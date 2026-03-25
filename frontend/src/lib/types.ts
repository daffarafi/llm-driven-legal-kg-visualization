/* TypeScript interfaces for the Legal KG Visualization app */

// --- Graph ---

export interface GraphNode {
  id: string;
  labels: string[];
  label?: string;
  content?: string;
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
  bab: DocumentSection[];
  pasal: DocumentSection[];
  ayat: DocumentSection[];
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
  UndangUndang: "#3b82f6", // blue
  Bab: "#8b5cf6",          // purple
  Bagian: "#a78bfa",       // light purple
  Pasal: "#22c55e",        // green
  Ayat: "#86efac",         // light green
  EntitasHukum: "#f97316", // orange
  PerbuatanHukum: "#ef4444", // red
  Sanksi: "#dc2626",       // dark red
  KonsepHukum: "#eab308",  // yellow
};

export const NODE_SIZES: Record<string, number> = {
  UndangUndang: 8,
  Bab: 5,
  Pasal: 4,
  Ayat: 3,
  EntitasHukum: 5,
  PerbuatanHukum: 4,
  Sanksi: 4,
  KonsepHukum: 4,
};
