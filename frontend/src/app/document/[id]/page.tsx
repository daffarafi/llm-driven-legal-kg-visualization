"use client";

import { useEffect, useState, use } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ArrowLeft, ChevronDown, ChevronRight, FileText } from "lucide-react";
import Link from "next/link";
import { getDocument } from "@/lib/api";
import { NODE_COLORS } from "@/lib/types";
import type { DocumentData, DocumentSection } from "@/lib/types";

/**
 * Format legal content: detect numbered/lettered list patterns and render vertically.
 * Handles nested lists recursively with custom indentation depth.
 */
function formatLegalContent(text: string, depth: number = 0) {
  if (!text) return <span className="text-muted-foreground">(isi tidak tersedia)</span>;

  // Pattern 1: Numbered items like "1. xxx 2. xxx 3. xxx"
  const numberedMatch = text.match(/(?:^|:\s*)\d+\.\s+/);
  // Pattern 2: Lettered items like "a. xxx; b. xxx"
  const letteredMatch = text.match(/(?:^|:\s*)[a-z]\.\s+/);

  if (!numberedMatch && !letteredMatch) return <>{text}</>;

  // Determine which list pattern appears first in the text (the outer-most list)
  let listMatch: RegExpMatchArray;
  let isNumbered = false;

  if (numberedMatch && letteredMatch) {
    const numIdx = text.indexOf(numberedMatch[0]);
    const letIdx = text.indexOf(letteredMatch[0]);
    if (numIdx < letIdx) {
      listMatch = numberedMatch;
      isNumbered = true;
    } else {
      listMatch = letteredMatch;
      isNumbered = false;
    }
  } else if (numberedMatch) {
    listMatch = numberedMatch;
    isNumbered = true;
  } else {
    listMatch = letteredMatch!;
    isNumbered = false;
  }

  const splitIdx = text.indexOf(listMatch[0]);
  const intro = text.substring(0, splitIdx).replace(/:\s*$/, "").trim();
  const listPart = text.substring(splitIdx).replace(/^:\s*/, "");

  let items: string[];

  if (isNumbered) {
    // Split on "N. " pattern (numbered definitions like Pasal 1)
    items = listPart
      .split(/(?<!\d)(?=\d+\.\s+)/)
      .map(s => s.trim())
      .filter(Boolean);
  } else {
    // Split on "; a." or "; dan a." or just "a. " at boundaries
    items = listPart
      .split(/;\s*(?:dan\s+|atau\s+)?(?=[a-z]\.\s)/i)
      .map(s => s.replace(/;\s*$/, "").replace(/\.\s*$/, "").trim())
      .filter(Boolean);
  }

  return (
    <>
      {intro && <p className={depth > 0 ? "mb-1" : "mb-1.5"}>{intro}:</p>}
      <div className={`space-y-1 ${depth > 0 ? "pl-4 border-l border-border/10" : "pl-1"}`}>
        {items.map((item, i) => {
          const marker = item.match(/^([a-z]\.|^\d+\.)\s*/i);
          const markerText = marker?.[1] || "";
          const itemText = marker ? item.substring(marker[0].length) : item;
          return (
            <div key={i} className="flex gap-2">
              <span className="text-muted-foreground shrink-0 w-6 text-right">{markerText}</span>
              <div className="flex-1">{formatLegalContent(itemText, depth + 1)}</div>
            </div>
          );
        })}
      </div>
    </>
  );
}

export default function DocumentViewerPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [doc, setDoc] = useState<DocumentData | null>(null);
  const [loading, setLoading] = useState(true);
  const [expandedBab, setExpandedBab] = useState<Set<string>>(new Set());
  const [expandedBagian, setExpandedBagian] = useState<Set<string>>(new Set());
  const [expandedPasal, setExpandedPasal] = useState<Set<string>>(new Set());

  useEffect(() => {
    setLoading(true);
    getDocument(id)
      .then((data) => setDoc(data as DocumentData))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [id]);

  const toggle = (set: Set<string>, key: string, setter: (s: Set<string>) => void) => {
    const next = new Set(set);
    if (next.has(key)) next.delete(key);
    else next.add(key);
    setter(next);
  };

  // Group Bagian by BAB
  const bagianByBab = doc?.bagian?.reduce<Record<string, DocumentSection[]>>((acc, bg) => {
    const key = bg.bab || "";
    if (!acc[key]) acc[key] = [];
    acc[key].push(bg);
    return acc;
  }, {}) || {};

  // Group Pasal by Bagian and BAB (for those under a Bagian)
  const pasalByBagianAndBab = doc?.pasal?.reduce<Record<string, DocumentSection[]>>((acc, p) => {
    if (p.bagian && p.bab) {
      const key = `${p.bab}||${p.bagian}`;
      if (!acc[key]) acc[key] = [];
      acc[key].push(p);
    }
    return acc;
  }, {}) || {};

  // Group Pasal by BAB (direct, no Bagian)
  const pasalDirectByBab = doc?.pasal?.reduce<Record<string, DocumentSection[]>>((acc, p) => {
    if (!p.bagian && p.bab) {
      if (!acc[p.bab]) acc[p.bab] = [];
      acc[p.bab].push(p);
    }
    return acc;
  }, {}) || {};

  // Ungrouped Pasal (no BAB, e.g. UU_19_2016)
  const ungroupedPasal = doc?.pasal?.filter(p => !p.bab && !p.bagian) || [];

  // Group Ayat by Pasal
  const ayatByPasal = doc?.ayat?.reduce<Record<string, DocumentSection[]>>((acc, a) => {
    const key = a.pasal || "";
    if (!acc[key]) acc[key] = [];
    acc[key].push(a);
    return acc;
  }, {}) || {};

  const renderAyat = (pasalLabel: string) => {
    const ayats = ayatByPasal[pasalLabel];
    if (!ayats?.length) return null;
    return (
      <div className="mt-2 space-y-1.5 pl-3 border-l-2 border-emerald-500/20">
        {ayats.map((ayat) => {
          const shortLabel = ayat.label?.replace(`${pasalLabel} `, "") || ayat.label;
          return (
            <div key={ayat.id || ayat.label} className="p-2.5 rounded-lg border border-border/20">
              <Badge
                variant="outline"
                className="text-[10px] mb-1.5"
                style={{ borderColor: NODE_COLORS["Ayat"], color: NODE_COLORS["Ayat"] }}
              >
                {shortLabel}
              </Badge>
              <div className="text-sm leading-relaxed text-foreground/80">
                {formatLegalContent(ayat.content || "")}
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  const renderPasal = (pasal: DocumentSection) => {
    const hasAyat = (ayatByPasal[pasal.label || ""]?.length || 0) > 0;

    return (
      <div key={pasal.id || pasal.label} className="rounded-lg border border-border/20">
        <button
          onClick={() => hasAyat && toggle(expandedPasal, pasal.label || "", setExpandedPasal)}
          className={`flex items-center gap-2 w-full text-left px-3 py-2.5 ${hasAyat ? "cursor-pointer group" : "cursor-default"}`}
        >
          {hasAyat ? (
            expandedPasal.has(pasal.label || "") ? (
              <ChevronDown className="h-4 w-4 text-muted-foreground" />
            ) : (
              <ChevronRight className="h-4 w-4 text-muted-foreground" />
            )
          ) : (
            <div className="w-4" />
          )}
          <span
            className="text-sm font-semibold"
            style={{ color: NODE_COLORS["Pasal"] }}
          >
            {pasal.label}
          </span>
          {hasAyat && (
            <span className="text-[10px] text-muted-foreground ml-auto">
              {ayatByPasal[pasal.label || ""]?.length} ayat
            </span>
          )}
        </button>
        {/* Show content only if no ayat */}
        {!hasAyat && (
          <div className="px-3 pb-3 -mt-1">
            <div className="text-sm leading-relaxed text-foreground/90 pl-6">
              {formatLegalContent(pasal.content || "")}
            </div>
          </div>
        )}
        {/* Show ayat when expanded */}
        {hasAyat && expandedPasal.has(pasal.label || "") && (
          <div className="px-3 pb-3">
            {renderAyat(pasal.label || "")}
          </div>
        )}
      </div>
    );
  };

  const renderBagian = (bagian: DocumentSection, babLabel: string) => {
    const key = `${babLabel}||${bagian.label}`;
    const pasals = pasalByBagianAndBab[key] || [];
    const uniqueExpandedKey = `${babLabel}||${bagian.label}`;
    return (
      <div key={bagian.id || bagian.label} className="ml-2 border-l-2 border-purple-400/20 pl-3">
        <button
          onClick={() => toggle(expandedBagian, uniqueExpandedKey, setExpandedBagian)}
          className="flex items-center gap-2 w-full text-left py-2 group"
        >
          {expandedBagian.has(uniqueExpandedKey) ? (
            <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
          ) : (
            <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
          )}
          <h3 className="text-sm font-semibold text-purple-300 group-hover:text-purple-200 transition-colors">
            {bagian.label}
          </h3>
          <span className="text-[10px] text-muted-foreground ml-auto">
            {pasals.length} pasal
          </span>
        </button>
        {expandedBagian.has(uniqueExpandedKey) && (
          <div className="space-y-2 pb-2">
            {pasals.map(renderPasal)}
          </div>
        )}
      </div>
    );
  };

  if (loading) {
    return <div className="flex items-center justify-center h-[calc(100vh-3.5rem)] text-muted-foreground">Loading document...</div>;
  }

  if (!doc) {
    return <div className="flex items-center justify-center h-[calc(100vh-3.5rem)] text-muted-foreground">Document not found</div>;
  }

  const hasBab = doc.bab && doc.bab.length > 0;

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
        {hasBab ? (
          doc.bab!.map((bab) => {
            const bagians = bagianByBab[bab.label || ""] || [];
            const directPasals = pasalDirectByBab[bab.label || ""] || [];
            const totalPasals = directPasals.length + bagians.reduce((sum, bg) => {
              const key = `${bab.label || ""}||${bg.label || ""}`;
              return sum + (pasalByBagianAndBab[key]?.length || 0);
            }, 0);

            return (
              <Card key={bab.id || bab.label} className="bg-card/50 border-border/40">
                <button
                  onClick={() => toggle(expandedBab, bab.label || "", setExpandedBab)}
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
                    {totalPasals} pasal
                  </span>
                </button>

                {expandedBab.has(bab.label || "") && (
                  <CardContent className="pt-0 pb-4 space-y-3">
                    {/* Bagian sections */}
                    {bagians.map((bg) => renderBagian(bg, bab.label || ""))}
                    {/* Direct Pasals (no Bagian) */}
                    {directPasals.map(renderPasal)}
                  </CardContent>
                )}
              </Card>
            );
          })
        ) : (
          /* Flat Pasal list (e.g. UU_19_2016) */
          <Card className="bg-card/50 border-border/40">
            <CardContent className="py-4 space-y-3">
              {ungroupedPasal.map(renderPasal)}
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
