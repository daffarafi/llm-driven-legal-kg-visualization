"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  Network,
  MessageSquare,
  BarChart3,
  ArrowRight,
  Database,
  GitBranch,
  Scale,
  FileText,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { getStats, getDocuments } from "@/lib/api";
import type { StatsData, Regulation } from "@/lib/types";

export default function HomePage() {
  const [stats, setStats] = useState<StatsData | null>(null);
  const [docCount, setDocCount] = useState<number>(0);

  useEffect(() => {
    getStats()
      .then((data) => setStats(data as StatsData))
      .catch(() => {});
    getDocuments()
      .then((data) => {
        const d = data as { regulations?: Regulation[] };
        setDocCount(d.regulations?.length || 0);
      })
      .catch(() => {});
  }, []);

  return (
    <div className="min-h-[calc(100vh-3.5rem)]">
      {/* Hero */}
      <section className="relative overflow-hidden border-b border-border/40">
        <div className="absolute inset-0 bg-gradient-to-br from-amber-500/5 via-transparent to-blue-500/5" />
        <div className="max-w-[1000px] mx-auto px-4 py-20 relative">
          <h1 className="text-4xl md:text-5xl font-bold tracking-tight mb-4">
            <span className="bg-gradient-to-r from-amber-500 to-orange-500 bg-clip-text text-transparent">
              Legal Knowledge Graph
            </span>
            <br />
            Visualization
          </h1>
          <p className="text-lg text-muted-foreground max-w-[600px] mb-8">
            Eksplorasi interaktif Knowledge Graph multi-dokumen hukum Indonesia.
            Meliputi {docCount > 0 ? `${docCount} peraturan` : "klaster peraturan"} ITE & Perlindungan Data Pribadi
            dengan pelacakan amandemen dan relasi antar-dokumen.
          </p>
          <div className="flex gap-3">
            <Link href="/explore">
              <Button size="lg" className="bg-amber-600 hover:bg-amber-700 text-white">
                <Network className="mr-2 h-4 w-4" /> Jelajahi Graph
              </Button>
            </Link>
            <Link href="/qa">
              <Button size="lg" variant="outline">
                <MessageSquare className="mr-2 h-4 w-4" /> Tanya AI
              </Button>
            </Link>
          </div>
        </div>
      </section>

      {/* Stats */}
      <section className="max-w-[1000px] mx-auto px-4 py-12">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-12">
          <Card className="bg-card/50 border-border/40">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-muted-foreground flex items-center gap-2">
                <Database className="h-4 w-4 text-amber-500" /> Total Nodes
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-3xl font-bold">{stats?.total_nodes?.toLocaleString() ?? "—"}</p>
            </CardContent>
          </Card>
          <Card className="bg-card/50 border-border/40">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-muted-foreground flex items-center gap-2">
                <GitBranch className="h-4 w-4 text-blue-500" /> Total Edges
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-3xl font-bold">{stats?.total_edges?.toLocaleString() ?? "—"}</p>
            </CardContent>
          </Card>
          <Card className="bg-card/50 border-border/40">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-muted-foreground flex items-center gap-2">
                <Scale className="h-4 w-4 text-cyan-500" /> Peraturan
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-3xl font-bold">{docCount || "—"}</p>
            </CardContent>
          </Card>
          <Card className="bg-card/50 border-border/40">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-muted-foreground flex items-center gap-2">
                <BarChart3 className="h-4 w-4 text-green-500" /> Node Types
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-3xl font-bold">{stats?.node_types?.length ?? "—"}</p>
            </CardContent>
          </Card>
        </div>

        {/* Feature cards */}
        <h2 className="text-2xl font-bold mb-6">Fitur</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {[
            {
              href: "/explore",
              icon: Network,
              title: "KG Explorer",
              desc: "Jelajahi Knowledge Graph visual — filter tipe node, warnai berdasarkan dokumen sumber.",
              color: "text-amber-500",
            },
            {
              href: "/qa",
              icon: MessageSquare,
              title: "QA Panel",
              desc: "Tanya jawab multi-dokumen — AI generate Cypher, eksekusi di KG, jawab dalam bahasa Indonesia.",
              color: "text-blue-500",
            },
            {
              href: "/analytics",
              icon: BarChart3,
              title: "Analytics",
              desc: "Dashboard statistik — distribusi node/edge, top entities, overview KG multi-dokumen.",
              color: "text-green-500",
            },
            {
              href: "/document",
              icon: FileText,
              title: "Regulasi",
              desc: "Jelajahi klaster regulasi ITE & PDP — relasi antar-dokumen, amandemen, versi pasal.",
              color: "text-cyan-500",
            },
          ].map((f) => (
            <Link key={f.href} href={f.href}>
              <Card className="bg-card/50 border-border/40 hover:border-amber-500/40 transition-all hover:shadow-lg hover:shadow-amber-500/5 h-full group">
                <CardHeader>
                  <f.icon className={`h-8 w-8 ${f.color} mb-2`} />
                  <CardTitle className="text-lg">{f.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground mb-3">{f.desc}</p>
                  <span className="text-sm text-amber-500 flex items-center gap-1 group-hover:gap-2 transition-all">
                    Buka <ArrowRight className="h-3 w-3" />
                  </span>
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}
