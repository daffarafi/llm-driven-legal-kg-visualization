"use client";

import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import dynamic from "next/dynamic";
import ReactMarkdown from "react-markdown";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Send, User, Bot, ChevronDown, ChevronRight,
  Search, CheckCircle2, AlertCircle, Loader2, Code,
  GitBranch,
} from "lucide-react";
import { askQuestion, getNodeDetail } from "@/lib/api";
import type { ChatMessage, QAResponse, QAProcessStep, NodeDetail } from "@/lib/types";
import { NODE_COLORS, NODE_SIZES } from "@/lib/types";

const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), { ssr: false });

function getNodeType(labels: string[] | undefined): string {
  if (!labels) return "";
  return labels.find((l) => l !== "Entity") || labels[0] || "";
}

/* ─── Process Steps (collapsible) ─── */
function ProcessSteps({ steps }: { steps: QAProcessStep[] }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="mb-3">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
      >
        {expanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        <Code className="h-3 w-3" />
        Proses ({steps.length} langkah)
      </button>
      {expanded && (
        <div className="mt-2 ml-4 border-l-2 border-border/40 pl-3 space-y-2">
          {steps.map((s) => (
            <div key={s.step} className="flex items-start gap-2 text-xs">
              {s.status === "done" ? (
                <CheckCircle2 className="h-3.5 w-3.5 text-green-500 mt-0.5 shrink-0" />
              ) : s.status === "error" ? (
                <AlertCircle className="h-3.5 w-3.5 text-red-500 mt-0.5 shrink-0" />
              ) : (
                <Loader2 className="h-3.5 w-3.5 text-muted-foreground animate-spin mt-0.5 shrink-0" />
              )}
              <div>
                <span className="font-medium">{s.label}</span>
                {s.detail && (
                  <pre className="mt-1 text-[11px] text-muted-foreground bg-background/50 rounded p-1.5 overflow-x-auto whitespace-pre-wrap">
                    {s.detail}
                  </pre>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ─── Answer Content (with Markdown rendering) ─── */
function AnswerContent({ text, references, onNodeSelect }: {
  text: string;
  references: string[];
  onNodeSelect?: (label: string) => void;
}) {
  // Custom renderer: make "Pasal X" clickable inside markdown
  const renderTextWithRefs = useCallback((text: string) => {
    const parts = text.split(/(Pasal\s+\d+(?:\s+ayat\s+\(\d+\))?)/gi);
    return parts.map((part, i) => {
      const isRef = /^Pasal\s+\d+/i.test(part);
      if (isRef) {
        return (
          <button
            key={i}
            onClick={() => onNodeSelect?.(part)}
            className="text-amber-500 font-medium hover:underline hover:text-amber-400 cursor-pointer transition-colors"
          >
            {part}
          </button>
        );
      }
      return <span key={i}>{part}</span>;
    });
  }, [onNodeSelect]);

  // Custom markdown components to inject Pasal reference handling
  const mdComponents = useMemo(() => ({
    p: ({ children, ...props }: any) => {
      return (
        <p className="mb-2 last:mb-0" {...props}>
          {processChildren(children)}
        </p>
      );
    },
    li: ({ children, ...props }: any) => {
      return (
        <li className="ml-4 mb-1" {...props}>
          {processChildren(children)}
        </li>
      );
    },
    strong: ({ children, ...props }: any) => (
      <strong className="font-semibold text-foreground" {...props}>{children}</strong>
    ),
    ul: ({ children, ...props }: any) => (
      <ul className="list-disc pl-4 mb-2" {...props}>{children}</ul>
    ),
    ol: ({ children, ...props }: any) => (
      <ol className="list-decimal pl-4 mb-2" {...props}>{children}</ol>
    ),
    h1: ({ children, ...props }: any) => (
      <h1 className="text-base font-bold mb-2 mt-3" {...props}>{children}</h1>
    ),
    h2: ({ children, ...props }: any) => (
      <h2 className="text-sm font-bold mb-1.5 mt-2" {...props}>{children}</h2>
    ),
    h3: ({ children, ...props }: any) => (
      <h3 className="text-sm font-semibold mb-1 mt-2" {...props}>{children}</h3>
    ),
  }), []);

  // Process children to make Pasal references clickable
  function processChildren(children: any): any {
    if (!children) return children;
    if (typeof children === "string") {
      return renderTextWithRefs(children);
    }
    if (Array.isArray(children)) {
      return children.map((child, i) => {
        if (typeof child === "string") return renderTextWithRefs(child);
        return child;
      });
    }
    return children;
  }

  return (
    <div>
      <div className="text-sm leading-relaxed prose-sm">
        <ReactMarkdown components={mdComponents}>{text}</ReactMarkdown>
      </div>
      {references.length > 0 && (
        <div className="mt-3 pt-2 border-t border-border/40">
          <p className="text-xs text-muted-foreground mb-1">📌 Referensi:</p>
          <div className="flex flex-wrap gap-1">
            {references.map((ref, i) => (
              <button
                key={i}
                onClick={() => onNodeSelect?.(ref)}
                className="inline-flex items-center rounded-full border border-amber-500/30 text-amber-500 text-[10px] px-2 py-0.5 hover:bg-amber-500/10 hover:border-amber-500/50 cursor-pointer transition-colors"
              >
                {ref}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/* ─── Live Side-Panel Graph ─── */
function LiveGraph({ graphNodes, graphEdges, selectTriggerRef }: {
  graphNodes: Map<string, { id: string; labels: string[]; label?: string }>;
  graphEdges: { source: string; target: string; type: string }[];
  selectTriggerRef: React.MutableRefObject<((label: string) => void) | null>;
}) {
  const [selectedDetail, setSelectedDetail] = useState<NodeDetail | null>(null);
  // Use refs for state that ForceGraph2D callbacks read — avoids re-render loops
  const selectedNodeIdRef = useRef<string | null>(null);
  const neighborIdsRef = useRef<Set<string>>(new Set());
  const fgRef = useRef<any>(null);
  const [, forceUpdate] = useState(0); // trigger re-render for detail panel only

  // Build graph data
  const graphData = useMemo(() => {
    const nodes = Array.from(graphNodes.values()).map((n) => {
      const nodeType = getNodeType(n.labels);
      return {
        id: n.id,
        label: n.label || "",
        nodeType,
        color: NODE_COLORS[nodeType] || "#888",
        val: NODE_SIZES[nodeType] || 3,
      };
    });
    const nodeIds = new Set(nodes.map(n => n.id));
    const links = graphEdges
      .filter(e => nodeIds.has(e.source) && nodeIds.has(e.target))
      .map((e) => ({
        source: e.source,
        target: e.target,
        label: e.type,
      }));
    return { nodes, links };
  }, [graphNodes, graphEdges]);

  const updateNeighborIds = useCallback((nodeId: string | null) => {
    if (!nodeId) {
      neighborIdsRef.current = new Set();
      return;
    }
    const ids = new Set<string>();
    for (const e of graphEdges) {
      const src = typeof e.source === "object" ? (e.source as any).id : e.source;
      const tgt = typeof e.target === "object" ? (e.target as any).id : e.target;
      if (src === nodeId) ids.add(tgt);
      if (tgt === nodeId) ids.add(src);
    }
    neighborIdsRef.current = ids;
  }, [graphEdges]);

  // Click node → toggle select + fetch detail
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handleNodeClick = useCallback(async (node: any) => {
    if (!node.id) return;
    if (node.id === selectedNodeIdRef.current) {
      // Deselect
      selectedNodeIdRef.current = null;
      neighborIdsRef.current = new Set();
      setSelectedDetail(null);
      forceUpdate(n => n + 1);
      return;
    }
    // Select
    selectedNodeIdRef.current = node.id;
    updateNeighborIds(node.id);
    forceUpdate(n => n + 1);
    try {
      const detail = (await getNodeDetail(node.id)) as NodeDetail;
      setSelectedDetail(detail);
    } catch (e) {
      console.error("getNodeDetail failed:", e);
    }
  }, [updateNeighborIds]);

  // Expose selectNodeByLabel for parent to call from AnswerContent clicks
  const selectNodeByLabel = useCallback((label: string) => {
    // Find matching node by label (case-insensitive, partial match)
    const normalizedLabel = label.trim().toLowerCase();
    for (const n of graphNodes.values()) {
      const nodeLabel = (n.label || "").toLowerCase();
      if (nodeLabel === normalizedLabel || nodeLabel.startsWith(normalizedLabel)) {
        handleNodeClick({ id: n.id });
        return;
      }
    }
  }, [graphNodes, handleNodeClick]);

  // Register the selectNodeByLabel function on the ref so parent can call it
  useEffect(() => {
    selectTriggerRef.current = selectNodeByLabel;
  }, [selectNodeByLabel, selectTriggerRef]);

  const handleDeselect = useCallback(() => {
    selectedNodeIdRef.current = null;
    neighborIdsRef.current = new Set();
    setSelectedDetail(null);
    forceUpdate(n => n + 1);
  }, []);

  // Paint with highlight — reads from refs, no state deps
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const paintNode = useCallback((node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
    const size = (node.val || 3) * 1.5;
    const color = node.color || "#888";
    const nodeType = node.nodeType || "";
    const selectedId = selectedNodeIdRef.current;
    const neighbors = neighborIdsRef.current;

    const isSelected = node.id === selectedId;
    const isNeighbor = selectedId ? neighbors.has(node.id) : false;

    // Glow for selected
    if (isSelected) {
      ctx.beginPath();
      ctx.arc(node.x, node.y, size + 4, 0, 2 * Math.PI);
      ctx.fillStyle = "rgba(255, 180, 50, 0.3)";
      ctx.fill();
    }
    if (isNeighbor) {
      ctx.beginPath();
      ctx.arc(node.x, node.y, size + 3, 0, 2 * Math.PI);
      ctx.fillStyle = "rgba(255, 255, 255, 0.15)";
      ctx.fill();
    }

    // Circle
    ctx.beginPath();
    ctx.arc(node.x, node.y, size, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();

    // Border
    if (isSelected) { ctx.strokeStyle = "#ffb432"; ctx.lineWidth = 2; }
    else if (isNeighbor) { ctx.strokeStyle = "rgba(255,255,255,0.6)"; ctx.lineWidth = 1.2; }
    else { ctx.strokeStyle = "rgba(255,255,255,0.3)"; ctx.lineWidth = 0.5; }
    ctx.stroke();

    // Label
    const showLabel = globalScale > 1.2 || isSelected || isNeighbor;
    if (showLabel) {
      const fontSize = Math.max(10 / globalScale, 1.5);
      const label = node.label || "";
      const displayLabel = label.length > 16 ? label.slice(0, 14) + "…" : label;
      ctx.font = `${fontSize}px Inter, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillStyle = isSelected ? "#ffb432" : isNeighbor ? "rgba(255,255,255,0.95)" : "rgba(255,255,255,0.8)";
      ctx.fillText(displayLabel, node.x, node.y + size + 1.5);
    }

    // Abbreviation
    if (size >= 4) {
      const abbr = nodeType.slice(0, 2).toUpperCase();
      const abbrFontSize = Math.max(size * 0.7, 2);
      ctx.font = `bold ${abbrFontSize}px Inter, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = "rgba(0,0,0,0.7)";
      ctx.fillText(abbr, node.x, node.y);
    }
  }, []);

  if (graphData.nodes.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center text-muted-foreground px-6">
        <GitBranch className="h-10 w-10 text-amber-500/20 mb-3" />
        <p className="text-sm font-medium mb-1">Knowledge Graph</p>
        <p className="text-xs">Graph akan muncul di sini saat Anda bertanya. Node dan relasi yang berkaitan dengan jawaban akan divisualisasikan secara live.</p>
      </div>
    );
  }

  const hasDetail = !!selectedDetail;

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-3 py-2 border-b border-border/40 flex items-center gap-2 shrink-0">
        <GitBranch className="h-3.5 w-3.5 text-amber-500" />
        <span className="text-xs font-semibold">Knowledge Graph</span>
        <span className="text-[10px] text-muted-foreground ml-auto">
          {graphData.nodes.length} nodes · {graphData.links.length} edges
        </span>
      </div>

      {/* Graph Canvas */}
      <div style={{ height: hasDetail ? "55%" : "100%", transition: "height 0.2s" }}>
        <ForceGraph2D
          ref={fgRef}
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
            const selId = selectedNodeIdRef.current;
            if (!selId) return "rgba(255,255,255,0.35)";
            const src = typeof link.source === "object" ? link.source.id : link.source;
            const tgt = typeof link.target === "object" ? link.target.id : link.target;
            if (src === selId || tgt === selId) return "rgba(255,180,50,0.8)";
            return "rgba(255,255,255,0.35)";
          }}
          linkWidth={(link: any) => {
            const selId = selectedNodeIdRef.current;
            if (!selId) return 1.2;
            const src = typeof link.source === "object" ? link.source.id : link.source;
            const tgt = typeof link.target === "object" ? link.target.id : link.target;
            if (src === selId || tgt === selId) return 2.5;
            return 1.2;
          }}
          linkDirectionalArrowLength={3.5}
          linkDirectionalArrowRelPos={0.9}
          linkDirectionalArrowColor={() => "rgba(255,200,100,0.5)"}
          linkLabel={(link: any) => link.label || ""}
          onNodeClick={handleNodeClick}
          onBackgroundClick={handleDeselect}
          onNodeDragEnd={(node: any) => {
            node.fx = node.x;
            node.fy = node.y;
          }}
          backgroundColor="transparent"
          width={typeof window !== "undefined" ? Math.min(window.innerWidth * 0.4, 500) : 450}
          height={typeof window !== "undefined" ? (selectedDetail ? (window.innerHeight - 56) * 0.55 : window.innerHeight - 56) : 600}
        />
      </div>

      {/* Detail Panel */}
      {selectedDetail && (
        <div className="border-t border-border/40 overflow-hidden" style={{ height: "45%" }}>
          <ScrollArea className="h-full">
            <div className="p-3">
              <div className="flex items-start justify-between mb-2">
                <div>
                  <h4 className="text-sm font-semibold">{String(selectedDetail.properties?.label || selectedDetail.properties?.name || selectedDetail.id)}</h4>
                  <div className="flex gap-1 mt-1">
                    {selectedDetail.labels?.filter((l: string) => l !== "Entity").map((l: string) => (
                      <Badge key={l} variant="outline" className="text-[10px] px-1" style={{
                        borderColor: NODE_COLORS[l] || "#888",
                        color: NODE_COLORS[l] || "#888",
                      }}>
                        {l}
                      </Badge>
                    ))}
                  </div>
                </div>
                <button onClick={handleDeselect} className="text-muted-foreground hover:text-foreground p-1">
                  <ChevronDown className="h-3 w-3" />
                </button>
              </div>

              {!!selectedDetail.properties?.content && (
                <p className="text-[11px] text-muted-foreground mb-2 line-clamp-3">
                  {String(selectedDetail.properties.content)}
                </p>
              )}

              {selectedDetail.outgoing?.length > 0 && (
                <div className="mb-2">
                  <p className="text-[10px] text-muted-foreground font-semibold mb-1">Outgoing ({selectedDetail.outgoing.length})</p>
                  <div className="space-y-0.5 max-h-24 overflow-y-auto">
                    {selectedDetail.outgoing.map((r, i) => (
                      <button
                        key={`out-${i}`}
                        onClick={() => r.target_id && handleNodeClick({ id: r.target_id })}
                        className="flex items-center gap-1 w-full text-left px-1.5 py-0.5 text-[11px] hover:bg-accent rounded truncate"
                      >
                        <span className="text-amber-500">→</span>
                        <span className="text-muted-foreground">{r.type}</span>
                        <span className="text-amber-500">→</span>
                        <span className="truncate">{r.target_label || r.target_id}</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {selectedDetail.incoming?.length > 0 && (
                <div>
                  <p className="text-[10px] text-muted-foreground font-semibold mb-1">Incoming ({selectedDetail.incoming.length})</p>
                  <div className="space-y-0.5 max-h-24 overflow-y-auto">
                    {selectedDetail.incoming.map((r, i) => (
                      <button
                        key={`in-${i}`}
                        onClick={() => r.source_id && handleNodeClick({ id: r.source_id })}
                        className="flex items-center gap-1 w-full text-left px-1.5 py-0.5 text-[11px] hover:bg-accent rounded truncate"
                      >
                        <span className="truncate">{r.source_label || r.source_id}</span>
                        <span className="text-amber-500">→</span>
                        <span className="text-muted-foreground">{r.type}</span>
                        <span className="text-amber-500">→</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </ScrollArea>
        </div>
      )}
    </div>
  );
}

/* ─── Main QA Page ─── */
export default function QAPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Accumulated graph data across all responses
  const [allGraphNodes, setAllGraphNodes] = useState<Map<string, { id: string; labels: string[]; label?: string }>>(new Map());
  const [allGraphEdges, setAllGraphEdges] = useState<{ source: string; target: string; type: string }[]>([]);
  const graphSelectRef = useRef<((label: string) => void) | null>(null);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = async () => {
    const q = input.trim();
    if (!q || loading) return;

    const userMsg: ChatMessage = { role: "user", content: q, timestamp: new Date() };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const data = (await askQuestion(q)) as QAResponse;
      const assistantMsg: ChatMessage = {
        role: "assistant",
        content: data.answer,
        qa_response: data,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMsg]);

      // Merge graph data from this response into accumulated graph
      if (data.graph && data.graph.nodes.length > 0) {
        setAllGraphNodes((prev) => {
          const next = new Map(prev);
          for (const n of data.graph!.nodes) {
            if (!next.has(n.id)) {
              next.set(n.id, n);
            }
          }
          return next;
        });
        setAllGraphEdges((prev) => {
          const existing = new Set(prev.map(e => `${e.source}→${e.target}→${e.type}`));
          const newEdges = data.graph!.edges.filter(
            e => !existing.has(`${e.source}→${e.target}→${e.type}`)
          );
          return [...prev, ...newEdges];
        });
      }
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant" as const,
          content: "Maaf, terjadi kesalahan saat memproses pertanyaan. Pastikan backend API berjalan.",
          timestamp: new Date(),
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-[calc(100vh-3.5rem)]">
      {/* Left: Chat Panel */}
      <div className="flex flex-col flex-1 min-w-0">
        <ScrollArea className="flex-1 px-4">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-center py-20">
              <Search className="h-12 w-12 text-amber-500/30 mb-4" />
              <h2 className="text-xl font-bold mb-2">Tanya Hukum Indonesia</h2>
              <p className="text-sm text-muted-foreground max-w-[400px]">
                Tanyakan apa saja tentang hukum Indonesia. Sistem akan mencari di Knowledge Graph
                dan memberikan jawaban dengan referensi pasal.
              </p>
              <div className="flex flex-wrap gap-2 mt-6 justify-center">
                {[
                  "Apa sanksi pencemaran nama baik di UU ITE?",
                  "Pasal apa saja di Bab VII UU ITE?",
                  "Apa itu informasi elektronik menurut UU ITE?",
                ].map((q) => (
                  <button
                    key={q}
                    onClick={() => { setInput(q); }}
                    className="text-xs border border-border/40 rounded-full px-3 py-1.5 text-muted-foreground hover:text-foreground hover:border-amber-500/40 transition-colors"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="py-4 space-y-4">
            {messages.map((msg, i) => (
              <div key={i} className={`flex gap-3 ${msg.role === "user" ? "justify-end" : ""}`}>
                {msg.role === "assistant" && (
                  <div className="w-7 h-7 rounded-full bg-amber-500/10 flex items-center justify-center shrink-0 mt-1">
                    <Bot className="h-4 w-4 text-amber-500" />
                  </div>
                )}
                <div className={`max-w-[85%] ${
                  msg.role === "user"
                    ? "bg-amber-600 text-white rounded-2xl rounded-br-md px-4 py-2.5"
                    : "bg-card/50 border border-border/40 rounded-2xl rounded-bl-md px-4 py-3"
                }`}>
                  {msg.role === "assistant" && msg.qa_response?.process_steps && (
                    <ProcessSteps steps={msg.qa_response.process_steps} />
                  )}
                  {msg.role === "assistant" && msg.qa_response ? (
                    <AnswerContent text={msg.content} references={msg.qa_response.references} onNodeSelect={(label) => graphSelectRef.current?.(label)} />
                  ) : (
                    <p className="text-sm">{msg.content}</p>
                  )}
                </div>
                {msg.role === "user" && (
                  <div className="w-7 h-7 rounded-full bg-foreground/10 flex items-center justify-center shrink-0 mt-1">
                    <User className="h-4 w-4" />
                  </div>
                )}
              </div>
            ))}

            {loading && (
              <div className="flex gap-3">
                <div className="w-7 h-7 rounded-full bg-amber-500/10 flex items-center justify-center shrink-0">
                  <Bot className="h-4 w-4 text-amber-500" />
                </div>
                <div className="bg-card/50 border border-border/40 rounded-2xl rounded-bl-md px-4 py-3">
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Memproses pertanyaan...
                  </div>
                </div>
              </div>
            )}
            <div ref={scrollRef} />
          </div>
        </ScrollArea>

        {/* Input */}
        <div className="border-t border-border/40 p-4">
          <div className="flex gap-2 max-w-[700px] mx-auto">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSubmit()}
              placeholder="Ketik pertanyaan hukum..."
              className="flex-1"
              disabled={loading}
            />
            <Button onClick={handleSubmit} disabled={loading || !input.trim()} className="bg-amber-600 hover:bg-amber-700">
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>

      {/* Right: Live Knowledge Graph Panel */}
      <div className="w-[40%] max-w-[500px] border-l border-border/40 bg-card/20">
        <LiveGraph graphNodes={allGraphNodes} graphEdges={allGraphEdges} selectTriggerRef={graphSelectRef} />
      </div>
    </div>
  );
}
