"use client";

import { useEffect, useState, use } from "react";
import dynamic from "next/dynamic";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Button } from "@/components/ui/button";
import { ArrowLeft, ChevronDown, ChevronRight, FileText, Network } from "lucide-react";
import Link from "next/link";
import { getDocument, getNodeSubgraph } from "@/lib/api";
import { NODE_COLORS } from "@/lib/types";
import type { DocumentData, DocumentSection, GraphNode, GraphEdge } from "@/lib/types";

const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), { ssr: false });

export default function DocumentViewerPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [doc, setDoc] = useState<DocumentData | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedPasal, setSelectedPasal] = useState<DocumentSection | null>(null);
  const [subgraph, setSubgraph] = useState<{ nodes: GraphNode[]; edges: GraphEdge[] } | null>(null);
  const [expandedBab, setExpandedBab] = useState<Set<string>>(new Set());

  useEffect(() => {
    setLoading(true);
    getDocument(id)
      .then((data) => setDoc(data as DocumentData))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [id]);

  // Load subgraph when pasal is selected
  useEffect(() => {
    if (!selectedPasal?.id) {
      setSubgraph(null);
      return;
    }
    getNodeSubgraph(selectedPasal.id, 1)
      .then((data) => setSubgraph(data as { nodes: GraphNode[]; edges: GraphEdge[] }))
      .catch(console.error);
  }, [selectedPasal]);

  const toggleBab = (babLabel: string) => {
    setExpandedBab((prev) => {
      const next = new Set(prev);
      if (next.has(babLabel)) next.delete(babLabel);
      else next.add(babLabel);
      return next;
    });
  };

  // Group pasals by bab
  const pasalsByBab = doc?.pasal?.reduce<Record<string, DocumentSection[]>>((acc, p) => {
    const key = p.bab || "Lainnya";
    if (!acc[key]) acc[key] = [];
    acc[key].push(p);
    return acc;
  }, {}) || {};

  if (loading) {
    return <div className="flex items-center justify-center h-[calc(100vh-3.5rem)] text-muted-foreground">Loading document...</div>;
  }

  if (!doc) {
    return <div className="flex items-center justify-center h-[calc(100vh-3.5rem)] text-muted-foreground">Document not found</div>;
  }

  const graphData = subgraph ? {
    nodes: subgraph.nodes.map((n) => ({
      ...n,
      color: NODE_COLORS[n.labels?.[0] || ""] || "#888",
      val: n.id === selectedPasal?.id ? 8 : 4,
    })),
    links: subgraph.edges.map((e) => ({ source: e.source, target: e.target, label: e.type })),
  } : { nodes: [], links: [] };

  return (
    <div className="flex h-[calc(100vh-3.5rem)]">
      {/* Left: Document text */}
      <div className="flex-1 border-r border-border/40">
        <div className="border-b border-border/40 px-4 py-3 flex items-center gap-3">
          <Link href="/document">
            <Button size="sm" variant="ghost">
              <ArrowLeft className="h-4 w-4" />
            </Button>
          </Link>
          <FileText className="h-5 w-5 text-amber-500" />
          <h1 className="font-bold truncate">{doc.document?.label as string || "Dokumen"}</h1>
        </div>

        <ScrollArea className="h-[calc(100vh-7rem)]">
          <div className="p-6 max-w-[700px]">
            {doc.bab?.map((bab) => (
              <div key={bab.id || bab.label} className="mb-4">
                <button
                  onClick={() => toggleBab(bab.label || "")}
                  className="flex items-center gap-2 w-full text-left group"
                >
                  {expandedBab.has(bab.label || "") ? (
                    <ChevronDown className="h-4 w-4 text-muted-foreground" />
                  ) : (
                    <ChevronRight className="h-4 w-4 text-muted-foreground" />
                  )}
                  <h2 className="text-lg font-bold text-purple-400 group-hover:text-purple-300 transition-colors">
                    {bab.label}
                  </h2>
                </button>

                {expandedBab.has(bab.label || "") && (
                  <div className="ml-6 mt-2 space-y-3">
                    {(pasalsByBab[bab.label || ""] || []).map((pasal) => (
                      <div
                        key={pasal.id || pasal.label}
                        className={`p-3 rounded-lg border transition-all cursor-pointer ${
                          selectedPasal?.id === pasal.id
                            ? "border-amber-500/50 bg-amber-500/5"
                            : "border-border/20 hover:border-border/40"
                        }`}
                        onClick={() => setSelectedPasal(pasal)}
                      >
                        <div className="flex items-center gap-2 mb-1">
                          <Badge
                            variant="outline"
                            className="text-[10px]"
                            style={{ borderColor: NODE_COLORS["Pasal"], color: NODE_COLORS["Pasal"] }}
                          >
                            {pasal.label}
                          </Badge>
                          {selectedPasal?.id === pasal.id && (
                            <Network className="h-3 w-3 text-amber-500" />
                          )}
                        </div>
                        <p className="text-sm leading-relaxed text-foreground/90">
                          {pasal.content || "(isi tidak tersedia)"}
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}

            {/* Ungrouped pasals */}
            {pasalsByBab["Lainnya"]?.length > 0 && (
              <div className="mt-6">
                <h2 className="text-lg font-bold text-muted-foreground mb-3">Lainnya</h2>
                {pasalsByBab["Lainnya"].map((pasal) => (
                  <div
                    key={pasal.id || pasal.label}
                    className={`p-3 rounded-lg border mb-3 cursor-pointer transition-all ${
                      selectedPasal?.id === pasal.id
                        ? "border-amber-500/50 bg-amber-500/5"
                        : "border-border/20 hover:border-border/40"
                    }`}
                    onClick={() => setSelectedPasal(pasal)}
                  >
                    <Badge variant="outline" className="text-[10px] mb-1" style={{ borderColor: NODE_COLORS["Pasal"] }}>
                      {pasal.label}
                    </Badge>
                    <p className="text-sm leading-relaxed">{pasal.content || "(isi tidak tersedia)"}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        </ScrollArea>
      </div>

      {/* Right: Subgraph */}
      <div className="w-[400px] flex flex-col">
        <div className="border-b border-border/40 px-4 py-3">
          <h3 className="text-sm font-semibold flex items-center gap-2">
            <Network className="h-4 w-4 text-amber-500" />
            {selectedPasal ? `Subgraph: ${selectedPasal.label}` : "Pilih pasal untuk melihat subgraph"}
          </h3>
        </div>

        {selectedPasal && subgraph ? (
          <div className="flex-1">
            <ForceGraph2D
              graphData={graphData}
              nodeLabel={(node: any) => `${node.label || node.id}`}
              nodeColor={(node: any) => node.color || "#888"}
              nodeVal={(node: any) => node.val || 3}
              linkLabel={(link: any) => link.label || ""}
              linkColor={() => "rgba(255,255,255,0.15)"}
              linkDirectionalArrowLength={3}
              linkDirectionalArrowRelPos={1}
              backgroundColor="transparent"
              width={400}
              height={typeof window !== "undefined" ? window.innerHeight - 56 - 48 : 500}
            />
          </div>
        ) : (
          <div className="flex-1 flex items-center justify-center text-muted-foreground text-sm p-8 text-center">
            <div>
              <Network className="h-12 w-12 mx-auto mb-3 opacity-20" />
              <p>Klik pada pasal di sebelah kiri untuk melihat subgraph dan relasi terkait.</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
