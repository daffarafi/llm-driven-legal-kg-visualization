"use client";

import { useEffect, useState, useCallback, useMemo } from "react";
import dynamic from "next/dynamic";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Search, X, ExternalLink, Filter } from "lucide-react";
import { getGraph, getNodeDetail, searchNodes, getDocuments } from "@/lib/api";
import { NODE_COLORS, NODE_SIZES, DOC_COLORS } from "@/lib/types";
import type { GraphNode, GraphEdge, NodeDetail, SearchResult, Regulation } from "@/lib/types";

// Dynamic import for react-force-graph (SSR incompatible)
const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), { ssr: false });

// All available node types
const ALL_NODE_TYPES = [
  "Regulasi", "Bab", "Bagian", "Pasal", "Ayat",
  "EntitasHukum", "PerbuatanHukum", "Sanksi", "KonsepHukum",
];

export default function ExplorePage() {
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [edges, setEdges] = useState<GraphEdge[]>([]);
  const [selectedNode, setSelectedNode] = useState<NodeDetail | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [activeTypes, setActiveTypes] = useState<Set<string>>(new Set(ALL_NODE_TYPES));
  const [showFilters, setShowFilters] = useState(true);
  const [loading, setLoading] = useState(true);

  // Document filter
  const [availableDocs, setAvailableDocs] = useState<Regulation[]>([]);
  const [activeDocIds, setActiveDocIds] = useState<Set<string>>(new Set());

  // Compute neighbor IDs for the selected node
  const neighborIds = useMemo(() => {
    if (!selectedNodeId) return new Set<string>();
    const ids = new Set<string>();
    for (const e of edges) {
      const src = typeof e.source === "object" ? (e.source as any).id : e.source;
      const tgt = typeof e.target === "object" ? (e.target as any).id : e.target;
      if (src === selectedNodeId) ids.add(tgt);
      if (tgt === selectedNodeId) ids.add(src);
    }
    return ids;
  }, [selectedNodeId, edges]);


  // Load initial graph + documents
  useEffect(() => {
    setLoading(true);
    Promise.all([
      getGraph({ limit: 200 }),
      getDocuments(),
    ])
      .then(([graphRaw, docsRaw]) => {
        const d = graphRaw as { nodes: GraphNode[]; edges: GraphEdge[] };
        setNodes(d.nodes || []);
        setEdges(d.edges || []);

        const dd = docsRaw as { regulations?: Regulation[] };
        const regs = dd.regulations || [];
        setAvailableDocs(regs);
        setActiveDocIds(new Set(regs.map((r) => r.source_document_id || r.doc_id)));
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  // Helper: get the meaningful node type (skip "Entity" label)
  const getNodeType = useCallback((labels: string[] | undefined): string => {
    if (!labels) return "";
    return labels.find((l) => l !== "Entity") || labels[0] || "";
  }, []);

  // Document filter toggle
  const toggleDoc = (docId: string) => {
    setActiveDocIds((prev) => {
      const next = new Set(prev);
      if (next.has(docId)) {
        if (next.size > 1) next.delete(docId);
      } else {
        next.add(docId);
      }
      return next;
    });
  };

  // Filter graph data by node type AND document
  const graphData = useMemo(() => {
    const filteredNodes = nodes.filter((n) => {
      const typeMatch = n.labels?.some((l) => l !== "Entity" && activeTypes.has(l));
      const docMatch = activeDocIds.size === 0 || !n.source_document_id || activeDocIds.has(n.source_document_id);
      return typeMatch && docMatch;
    });
    const nodeIds = new Set(filteredNodes.map((n) => n.id));
    const filteredEdges = edges.filter(
      (e) => nodeIds.has(e.source) && nodeIds.has(e.target)
    );
    return {
      nodes: filteredNodes.map((n) => {
        const nodeType = getNodeType(n.labels);
        return {
          ...n,
          nodeType,
          color: NODE_COLORS[nodeType] || "#888",
          val: NODE_SIZES[nodeType] || 3,
        };
      }),
      links: filteredEdges.map((e) => ({
        source: e.source,
        target: e.target,
        label: e.type,
      })),
    };
  }, [nodes, edges, activeTypes, activeDocIds, getNodeType]);

  // Handle node click — toggle: click same node = deselect
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handleNodeClick = useCallback(async (node: any) => {
    if (!node.id) return;
    // Toggle off if clicking the same node
    if (node.id === selectedNodeId) {
      setSelectedNodeId(null);
      setSelectedNode(null);
      return;
    }
    setSelectedNodeId(node.id);
    try {
      const detail = (await getNodeDetail(node.id)) as NodeDetail;
      setSelectedNode(detail);
    } catch (e) {
      console.error(e);
    }
  }, [selectedNodeId]);

  // Search
  const handleSearch = useCallback(async () => {
    if (!searchQuery.trim()) {
      setSearchResults([]);
      return;
    }
    try {
      const data = (await searchNodes(searchQuery)) as {
        results: SearchResult[];
      };
      setSearchResults(data.results || []);
    } catch (e) {
      console.error(e);
    }
  }, [searchQuery]);

  // Toggle node type filter
  const toggleType = (type: string) => {
    setActiveTypes((prev) => {
      const next = new Set(prev);
      if (next.has(type)) next.delete(type);
      else next.add(type);
      return next;
    });
  };

  // Custom node canvas rendering — colored circle + text label + neighbor highlight
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const paintNode = useCallback((node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
    const label = node.label || "";
    const nodeType = node.nodeType || "";
    const color = node.color || NODE_COLORS[nodeType] || "#888";
    const size = (node.val || 3) * 1.5;
    const fontSize = Math.max(10 / globalScale, 1.5);

    const isSelected = node.id === selectedNodeId;
    const isNeighbor = selectedNodeId ? neighborIds.has(node.id) : false;

    // Glow effect for selected node
    if (isSelected) {
      ctx.beginPath();
      ctx.arc(node.x, node.y, size + 4, 0, 2 * Math.PI);
      ctx.fillStyle = "rgba(255, 180, 50, 0.3)";
      ctx.fill();
    }

    // Glow for neighbor nodes
    if (isNeighbor) {
      ctx.beginPath();
      ctx.arc(node.x, node.y, size + 3, 0, 2 * Math.PI);
      ctx.fillStyle = "rgba(255, 255, 255, 0.15)";
      ctx.fill();
    }

    // Draw node circle — always full color, no dimming
    ctx.beginPath();
    ctx.arc(node.x, node.y, size, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();

    // Border
    if (isSelected) {
      ctx.strokeStyle = "#ffb432";
      ctx.lineWidth = 2;
    } else if (isNeighbor) {
      ctx.strokeStyle = "rgba(255,255,255,0.6)";
      ctx.lineWidth = 1.2;
    } else {
      ctx.strokeStyle = "rgba(255,255,255,0.3)";
      ctx.lineWidth = 0.5;
    }
    ctx.stroke();

    // Draw label text (when zoomed in, or always for selected/neighbors)
    const showLabel = globalScale > 1.2 || isSelected || isNeighbor;
    if (showLabel) {
      const displayLabel = label.length > 20 ? label.slice(0, 18) + "…" : label;
      ctx.font = `${fontSize}px Inter, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillStyle = isSelected ? "#ffb432" : isNeighbor ? "rgba(255,255,255,0.95)" : "rgba(255,255,255,0.8)";
      ctx.fillText(displayLabel, node.x, node.y + size + 2);
    }

    // Type abbreviation inside node
    if (size >= 4) {
      const abbr = nodeType.slice(0, 2).toUpperCase();
      const abbrFontSize = Math.max(size * 0.7, 2);
      ctx.font = `bold ${abbrFontSize}px Inter, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = "rgba(0,0,0,0.7)";
      ctx.fillText(abbr, node.x, node.y);
    }
  }, [selectedNodeId, neighborIds]);

  return (
    <div className="flex h-[calc(100vh-3.5rem)]">
      {/* Left: Filters */}
      {showFilters && (
        <div className="w-56 border-r border-border/40 bg-card/30 p-3 flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold flex items-center gap-1.5">
              <Filter className="h-3.5 w-3.5" /> Filters
            </h3>
            <Button size="sm" variant="ghost" className="h-6 w-6 p-0" onClick={() => setShowFilters(false)}>
              <X className="h-3 w-3" />
            </Button>
          </div>

          {/* Search */}
          <div className="flex gap-1">
            <Input
              placeholder="Cari node..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              className="h-8 text-xs"
            />
            <Button size="sm" variant="ghost" className="h-8 w-8 p-0" onClick={handleSearch}>
              <Search className="h-3.5 w-3.5" />
            </Button>
          </div>

          {searchResults.length > 0 && (
            <ScrollArea className="max-h-32">
              {searchResults.map((r) => {
                const rType = getNodeType(r.labels);
                return (
                  <button
                    key={r.id}
                    onClick={() => handleNodeClick({ id: r.id })}
                    className="w-full text-left px-2 py-1 text-xs hover:bg-accent rounded truncate"
                  >
                    <Badge variant="outline" className="mr-1 text-[10px] px-1" style={{
                      borderColor: NODE_COLORS[rType] || "#888",
                      color: NODE_COLORS[rType] || "#888",
                    }}>
                      {rType.slice(0, 3)}
                    </Badge>
                    {r.label}
                  </button>
                );
              })}
            </ScrollArea>
          )}

          <Separator />

          {/* Document filter */}
          {availableDocs.length > 0 && (
            <>
              <h4 className="text-xs text-muted-foreground">Documents</h4>
              <div className="flex flex-col gap-1">
                {availableDocs.map((doc) => {
                  const docId = doc.source_document_id || doc.doc_id;
                  const isActive = activeDocIds.has(docId);
                  const shortName = doc.short_name || doc.label?.split(' tentang ')[0] || docId;
                  return (
                    <button
                      key={docId}
                      onClick={() => toggleDoc(docId)}
                      className={`flex items-center gap-2 px-2 py-1 text-xs rounded transition-colors ${
                        isActive ? "bg-accent" : "opacity-40"
                      }`}
                    >
                      <span
                        className="w-2.5 h-2.5 rounded-full"
                        style={{ backgroundColor: DOC_COLORS[docId] || "#888" }}
                      />
                      <span className="truncate">{shortName}</span>
                    </button>
                  );
                })}
              </div>
              <Separator />
            </>
          )}

          {/* Node type toggles */}
          <h4 className="text-xs text-muted-foreground">Node Types</h4>
          <div className="flex flex-col gap-1">
            {ALL_NODE_TYPES.map((type) => (
              <button
                key={type}
                onClick={() => toggleType(type)}
                className={`flex items-center gap-2 px-2 py-1 text-xs rounded transition-colors ${
                  activeTypes.has(type) ? "bg-accent" : "opacity-40"
                }`}
              >
                <span
                  className="w-2.5 h-2.5 rounded-full"
                  style={{ backgroundColor: NODE_COLORS[type] || "#888" }}
                />
                {type}
              </button>
            ))}
          </div>

          <Separator />
          <div className="text-[10px] text-muted-foreground">
            {graphData.nodes.length} nodes · {graphData.links.length} edges
          </div>
        </div>
      )}

      {/* Center: Graph Canvas */}
      <div className="flex-1 relative">
        {!showFilters && (
          <Button
            size="sm"
            variant="ghost"
            className="absolute top-2 left-2 z-10"
            onClick={() => setShowFilters(true)}
          >
            <Filter className="h-4 w-4" />
          </Button>
        )}
        {loading ? (
          <div className="flex items-center justify-center h-full text-muted-foreground">
            Loading graph...
          </div>
        ) : (
          <ForceGraph2D
            graphData={graphData}
            nodeCanvasObject={paintNode}
            nodePointerAreaPaint={(node: any, color: string, ctx: CanvasRenderingContext2D) => {
              const size = (node.val || 3) * 1.5;
              ctx.beginPath();
              ctx.arc(node.x, node.y, size + 2, 0, 2 * Math.PI);
              ctx.fillStyle = color;
              ctx.fill();
            }}
            linkColor={(link: any) => {
              if (!selectedNodeId) return "rgba(255,255,255,0.35)";
              const src = typeof link.source === "object" ? link.source.id : link.source;
              const tgt = typeof link.target === "object" ? link.target.id : link.target;
              if (src === selectedNodeId || tgt === selectedNodeId) return "rgba(255,180,50,0.8)";
              return "rgba(255,255,255,0.35)"; // normal brightness
            }}
            linkWidth={(link: any) => {
              if (!selectedNodeId) return 1.2;
              const src = typeof link.source === "object" ? link.source.id : link.source;
              const tgt = typeof link.target === "object" ? link.target.id : link.target;
              if (src === selectedNodeId || tgt === selectedNodeId) return 2.5;
              return 1.2; // normal width
            }}
            linkDirectionalArrowLength={4}
            linkDirectionalArrowRelPos={0.9}
            linkDirectionalArrowColor={() => "rgba(255,200,100,0.6)"}
            linkLabel={(link: any) => link.label || ""}
            onNodeClick={handleNodeClick}
            onBackgroundClick={() => {
              setSelectedNodeId(null);
              setSelectedNode(null);
            }}
            onNodeDragEnd={(node: any) => {
              node.fx = node.x;
              node.fy = node.y;
            }}
            backgroundColor="transparent"
            width={typeof window !== "undefined" ? window.innerWidth - (showFilters ? 224 : 0) - (selectedNode ? 350 : 0) : 800}
            height={typeof window !== "undefined" ? window.innerHeight - 56 : 600}
          />
        )}
      </div>

      {/* Right: Node Detail */}
      {selectedNode && (
        <div className="w-[350px] border-l border-border/40 bg-card/30">
          <ScrollArea className="h-[calc(100vh-3.5rem)]">
            <div className="p-4">
              <div className="flex items-start justify-between mb-3">
                <div>
                  <Badge style={{
                    backgroundColor: NODE_COLORS[selectedNode.labels?.[0] || ""] || "#888",
                    color: "#fff",
                  }}>
                    {selectedNode.labels?.[0]}
                  </Badge>
                  <h2 className="text-lg font-bold mt-1">
                    {(selectedNode.properties?.label as string) || selectedNode.id}
                  </h2>
                </div>
                <Button size="sm" variant="ghost" className="h-6 w-6 p-0" onClick={() => setSelectedNode(null)}>
                  <X className="h-3 w-3" />
                </Button>
              </div>

              {/* Content */}
              {Boolean(selectedNode.properties?.content) && (
                <Card className="mb-3 bg-background/50">
                  <CardHeader className="pb-1 pt-3 px-3">
                    <CardTitle className="text-xs text-muted-foreground">Isi</CardTitle>
                  </CardHeader>
                  <CardContent className="px-3 pb-3">
                    <p className="text-sm leading-relaxed">
                      {String(selectedNode.properties.content).slice(0, 500)}
                      {String(selectedNode.properties.content).length > 500 && "..."}
                    </p>
                  </CardContent>
                </Card>
              )}

              {/* Relations */}
              {selectedNode.outgoing?.length > 0 && (
                <div className="mb-3">
                  <h3 className="text-xs text-muted-foreground mb-1.5">Outgoing Relations</h3>
                  {selectedNode.outgoing.map((r, i) => (
                    <button
                      key={`out-${i}`}
                      onClick={() => r.target_id && handleNodeClick({ id: r.target_id })}
                      className="flex items-center gap-1.5 w-full text-left px-2 py-1 text-xs hover:bg-accent rounded"
                    >
                      <span className="text-amber-500">→</span>
                      <span className="text-muted-foreground">{r.type}</span>
                      <span className="text-amber-500">→</span>
                      <span className="truncate">{r.target_label || r.target_id}</span>
                    </button>
                  ))}
                </div>
              )}

              {selectedNode.incoming?.length > 0 && (
                <div className="mb-3">
                  <h3 className="text-xs text-muted-foreground mb-1.5">Incoming Relations</h3>
                  {selectedNode.incoming.map((r, i) => (
                    <button
                      key={`in-${i}`}
                      onClick={() => r.source_id && handleNodeClick({ id: r.source_id })}
                      className="flex items-center gap-1.5 w-full text-left px-2 py-1 text-xs hover:bg-accent rounded"
                    >
                      <span className="truncate">{r.source_label || r.source_id}</span>
                      <span className="text-blue-500">→</span>
                      <span className="text-muted-foreground">{r.type}</span>
                      <span className="text-blue-500">→</span>
                    </button>
                  ))}
                </div>
              )}

              {/* Document button - only for Regulasi nodes */}
              {selectedNode.labels?.some(l => l === "Regulasi") && (
                <>
                  <Separator className="my-3" />
                  <a href={`/document/${encodeURIComponent(selectedNode.id)}`}>
                    <Button size="sm" variant="outline" className="w-full">
                      <ExternalLink className="h-3 w-3 mr-1" /> Lihat Dokumen
                    </Button>
                  </a>
                </>
              )}
            </div>
          </ScrollArea>
        </div>
      )}
    </div>
  );
}
