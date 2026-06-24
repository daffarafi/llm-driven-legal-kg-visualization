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
  bagian?: string;
  pasal?: string;
}

export interface DocumentData {
  document: Record<string, unknown>;
  bab?: DocumentSection[];
  bagian?: DocumentSection[];
  pasal?: DocumentSection[];
  ayat?: DocumentSection[];
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

export function getNodeColor(label: string, isDark: boolean = true): string {
  const baseColor = NODE_COLORS[label] || "#888";
  if (!isDark) {
    switch (label) {
      case "Ayat":
        return "#15803d"; // Darker green (green-700) for readability
      case "Pasal":
        return "#16a34a"; // Darker green (green-600)
      case "Bagian":
        return "#7c3aed"; // Darker purple (violet-600)
      case "KonsepHukum":
        return "#b45309"; // Darker amber (amber-700)
      case "EntitasHukum":
        return "#ea580c"; // Darker orange (orange-600)
      case "VersiPasal":
        return "#0d9488"; // Darker teal (teal-600)
      default:
        return baseColor;
    }
  }
  return baseColor;
}

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
  POJK_11_2022: "#06b6d4",
};
