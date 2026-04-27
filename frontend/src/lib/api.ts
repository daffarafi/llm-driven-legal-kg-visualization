/* API client for Legal KG backend */

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

async function fetchAPI<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });
  if (!res.ok) {
    throw new Error(`API Error ${res.status}: ${res.statusText}`);
  }
  return res.json();
}

// --- Graph ---

export async function getGraph(params?: {
  types?: string;
  relations?: string;
  doc_ids?: string;
  limit?: number;
}) {
  const searchParams = new URLSearchParams();
  if (params?.types) searchParams.set("types", params.types);
  if (params?.relations) searchParams.set("relations", params.relations);
  if (params?.doc_ids) searchParams.set("doc_ids", params.doc_ids);
  if (params?.limit) searchParams.set("limit", params.limit.toString());
  const qs = searchParams.toString();
  return fetchAPI(`/api/graph${qs ? `?${qs}` : ""}`);
}

export async function getNodeDetail(nodeId: string) {
  return fetchAPI(`/api/node/${encodeURIComponent(nodeId)}`);
}

export async function getNodeSubgraph(nodeId: string, depth = 1) {
  return fetchAPI(
    `/api/node/${encodeURIComponent(nodeId)}/subgraph?depth=${depth}`
  );
}

// --- Search ---

export async function searchNodes(query: string, mode = "keyword", limit = 20) {
  return fetchAPI(
    `/api/search?q=${encodeURIComponent(query)}&mode=${mode}&limit=${limit}`
  );
}

// --- QA ---

export async function askQuestion(question: string, doc_ids?: string[]) {
  return fetchAPI("/api/qa", {
    method: "POST",
    body: JSON.stringify({ question, doc_ids: doc_ids?.length ? doc_ids : null }),
  });
}

// --- Stats ---

export async function getStats(doc_id?: string) {
  const qs = doc_id ? `?doc_id=${encodeURIComponent(doc_id)}` : "";
  return fetchAPI(`/api/stats${qs}`);
}

// --- Documents ---

export async function getDocuments() {
  return fetchAPI("/api/documents");
}

export async function getDocument(docId: string) {
  return fetchAPI(`/api/document/${encodeURIComponent(docId)}`);
}

// --- Regulations (Multi-document) ---

export async function getRegulationGraph() {
  return fetchAPI("/api/regulations/graph");
}

export async function getAmendments() {
  return fetchAPI("/api/regulations/amendments");
}
