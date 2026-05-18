"use client";

import { useEffect, useState, use } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { ArrowLeft, ChevronDown, ChevronRight, FileText } from "lucide-react";
import Link from "next/link";
import { getDocument } from "@/lib/api";
import { NODE_COLORS } from "@/lib/types";
import type { DocumentData, DocumentSection } from "@/lib/types";

export default function DocumentViewerPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [doc, setDoc] = useState<DocumentData | null>(null);
  const [loading, setLoading] = useState(true);
  const [expandedBab, setExpandedBab] = useState<Set<string>>(new Set());

  useEffect(() => {
    setLoading(true);
    getDocument(id)
      .then((data) => setDoc(data as DocumentData))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [id]);

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

  return (
    <div className="max-w-[900px] mx-auto px-4 py-6">
      {/* Header */}
      <div className="flex items-center gap-3 mb-6">
        <Link href="/document">
          <Button size="sm" variant="ghost">
            <ArrowLeft className="h-4 w-4" />
          </Button>
        </Link>
        <FileText className="h-5 w-5 text-amber-500" />
        <h1 className="font-bold truncate">{doc.document?.label as string || "Dokumen"}</h1>
      </div>

      {/* Document structure */}
      <div className="space-y-4">
        {doc.bab?.map((bab) => (
          <Card key={bab.id || bab.label} className="bg-card/50 border-border/40">
            <button
              onClick={() => toggleBab(bab.label || "")}
              className="flex items-center gap-2 w-full text-left px-4 py-3 group"
            >
              {expandedBab.has(bab.label || "") ? (
                <ChevronDown className="h-4 w-4 text-muted-foreground" />
              ) : (
                <ChevronRight className="h-4 w-4 text-muted-foreground" />
              )}
              <h2 className="text-base font-bold text-purple-400 group-hover:text-purple-300 transition-colors">
                {bab.label}
              </h2>
              <span className="text-xs text-muted-foreground ml-auto">
                {(pasalsByBab[bab.label || ""] || []).length} pasal
              </span>
            </button>

            {expandedBab.has(bab.label || "") && (
              <CardContent className="pt-0 pb-4 space-y-3">
                {(pasalsByBab[bab.label || ""] || []).map((pasal) => (
                  <div
                    key={pasal.id || pasal.label}
                    className="p-3 rounded-lg border border-border/20"
                  >
                    <Badge
                      variant="outline"
                      className="text-[10px] mb-1.5"
                      style={{ borderColor: NODE_COLORS["Pasal"], color: NODE_COLORS["Pasal"] }}
                    >
                      {pasal.label}
                    </Badge>
                    <p className="text-sm leading-relaxed text-foreground/90">
                      {pasal.content || "(isi tidak tersedia)"}
                    </p>
                  </div>
                ))}
              </CardContent>
            )}
          </Card>
        ))}

        {/* Ungrouped pasals */}
        {pasalsByBab["Lainnya"]?.length > 0 && (
          <Card className="bg-card/50 border-border/40">
            <CardContent className="py-4 space-y-3">
              <h2 className="text-base font-bold text-muted-foreground">Lainnya</h2>
              {pasalsByBab["Lainnya"].map((pasal) => (
                <div
                  key={pasal.id || pasal.label}
                  className="p-3 rounded-lg border border-border/20"
                >
                  <Badge variant="outline" className="text-[10px] mb-1.5" style={{ borderColor: NODE_COLORS["Pasal"] }}>
                    {pasal.label}
                  </Badge>
                  <p className="text-sm leading-relaxed">{pasal.content || "(isi tidak tersedia)"}</p>
                </div>
              ))}
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
