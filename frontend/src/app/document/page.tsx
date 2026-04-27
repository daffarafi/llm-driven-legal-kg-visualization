"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Card, CardContent } from "@/components/ui/card";
import {
  FileText,
  ChevronRight,
  BookOpen,
  Scale,
  GitBranch,
  CheckCircle2,
  XCircle,
  AlertCircle,
} from "lucide-react";
import { getDocuments } from "@/lib/api";
import type { Regulation } from "@/lib/types";

const REG_TYPE_COLORS: Record<string, string> = {
  UU: "bg-blue-500/10 text-blue-400 border-blue-500/20",
  PP: "bg-green-500/10 text-green-400 border-green-500/20",
  POJK: "bg-cyan-500/10 text-cyan-400 border-cyan-500/20",
  Perpres: "bg-orange-500/10 text-orange-400 border-orange-500/20",
  Permen: "bg-yellow-500/10 text-yellow-400 border-yellow-500/20",
};

const STATUS_CONFIG: Record<string, { icon: typeof CheckCircle2; color: string }> = {
  berlaku: { icon: CheckCircle2, color: "text-green-400" },
  "berlaku (diamandemen)": { icon: AlertCircle, color: "text-amber-400" },
  dicabut: { icon: XCircle, color: "text-red-400" },
};

export default function DocumentsListPage() {
  const [regulations, setRegulations] = useState<Regulation[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getDocuments()
      .then((data) => {
        const d = data as { regulations?: Regulation[]; documents?: Regulation[] };
        setRegulations(d.regulations || []);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  // Group by type
  const grouped = regulations.reduce(
    (acc, reg) => {
      const type = reg.regulation_type || "Lainnya";
      (acc[type] = acc[type] || []).push(reg);
      return acc;
    },
    {} as Record<string, Regulation[]>
  );

  const typeOrder = ["UU", "POJK", "PP", "Perpres", "Permen", "Lainnya"];

  return (
    <div className="max-w-[900px] mx-auto px-4 py-8">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Scale className="h-6 w-6 text-amber-500" /> Klaster Regulasi ITE &
            PDP
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            {regulations.length} peraturan dalam Knowledge Graph — klik untuk
            eksplorasi
          </p>
        </div>
        <Link href="/document/regulations-graph">
          <div className="flex items-center gap-1 text-sm text-amber-500 hover:text-amber-400 transition-colors cursor-pointer">
            <GitBranch className="h-4 w-4" /> Lihat Relasi
          </div>
        </Link>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-20">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-amber-500" />
        </div>
      ) : regulations.length === 0 ? (
        <Card className="bg-card/50 border-border/40">
          <CardContent className="py-12 text-center">
            <BookOpen className="h-12 w-12 text-muted-foreground mx-auto mb-3" />
            <p className="text-muted-foreground">
              Belum ada Peraturan nodes di Neo4j.
              <br />
              Jalankan <code className="text-amber-500">batch_runner.py --load-edges</code> terlebih dahulu.
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-8">
          {typeOrder
            .filter((t) => grouped[t]?.length)
            .map((type) => (
              <div key={type}>
                <div className="flex items-center gap-2 mb-3">
                  <span
                    className={`px-2 py-0.5 rounded text-xs font-medium border ${
                      REG_TYPE_COLORS[type] || "bg-gray-500/10 text-gray-400 border-gray-500/20"
                    }`}
                  >
                    {type}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {grouped[type].length} dokumen
                  </span>
                </div>
                <div className="space-y-2">
                  {grouped[type].map((reg) => {
                    const statusConf =
                      STATUS_CONFIG[reg.status || ""] || STATUS_CONFIG["berlaku"];
                    const StatusIcon = statusConf.icon;
                    return (
                      <Link
                        key={reg.doc_id}
                        href={`/document/${encodeURIComponent(reg.doc_id)}`}
                      >
                        <Card className="bg-card/50 border-border/40 hover:border-amber-500/40 transition-all cursor-pointer group">
                          <CardContent className="flex items-center justify-between py-3 px-4">
                            <div className="flex items-center gap-3 min-w-0">
                              <FileText className="h-7 w-7 text-amber-500/60 shrink-0" />
                              <div className="min-w-0">
                                <div className="flex items-center gap-2">
                                  <p className="font-medium text-sm truncate">
                                    {reg.short_name || reg.label}
                                  </p>
                                  <StatusIcon
                                    className={`h-3.5 w-3.5 shrink-0 ${statusConf.color}`}
                                  />
                                </div>
                                <p className="text-xs text-muted-foreground truncate">
                                  {reg.label}
                                </p>
                              </div>
                            </div>
                            <div className="flex items-center gap-4 shrink-0">
                              <div className="text-right hidden sm:block">
                                <p className="text-xs text-muted-foreground">
                                  {reg.year}
                                </p>
                                <p className="text-xs text-muted-foreground">
                                  {reg.entity_count || 0} entitas
                                </p>
                              </div>
                              <ChevronRight className="h-4 w-4 text-muted-foreground group-hover:text-amber-500 transition-colors" />
                            </div>
                          </CardContent>
                        </Card>
                      </Link>
                    );
                  })}
                </div>
              </div>
            ))}
        </div>
      )}
    </div>
  );
}
