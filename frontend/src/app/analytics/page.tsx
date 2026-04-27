"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Database, GitBranch, Layers, TrendingUp, FileText } from "lucide-react";
import { getStats, getDocuments } from "@/lib/api";
import type { StatsData, TypeCount, Regulation } from "@/lib/types";
import { NODE_COLORS, DOC_COLORS } from "@/lib/types";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend,
} from "recharts";

const CHART_COLORS = [
  "#3b82f6", "#8b5cf6", "#22c55e", "#f97316",
  "#ef4444", "#eab308", "#06b6d4", "#ec4899",
];

export default function AnalyticsPage() {
  const [stats, setStats] = useState<StatsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [availableDocs, setAvailableDocs] = useState<Regulation[]>([]);
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null); // null = semua

  // Fetch documents on mount
  useEffect(() => {
    getDocuments()
      .then((data) => {
        const d = data as { regulations?: Regulation[] };
        setAvailableDocs(d.regulations || []);
      })
      .catch(() => {});
  }, []);

  // Fetch stats (re-fetch when selectedDocId changes)
  useEffect(() => {
    setLoading(true);
    getStats(selectedDocId || undefined)
      .then((data) => setStats(data as StatsData))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [selectedDocId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-[calc(100vh-3.5rem)] text-muted-foreground">
        Loading analytics...
      </div>
    );
  }

  if (!stats) {
    return (
      <div className="flex items-center justify-center h-[calc(100vh-3.5rem)] text-muted-foreground">
        Tidak dapat memuat data. Pastikan backend dan Neo4j berjalan.
      </div>
    );
  }

  return (
    <div className="max-w-[1100px] mx-auto px-4 py-8">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Analytics Dashboard</h1>
      </div>

      {/* Document selector */}
      {availableDocs.length > 0 && (
        <div className="flex items-center gap-2 mb-6">
          <span className="text-xs text-muted-foreground uppercase tracking-wider">Filter:</span>
          <button
            onClick={() => setSelectedDocId(null)}
            className={`text-xs px-3 py-1 rounded-full border transition-all ${
              selectedDocId === null
                ? "bg-amber-500/15 border-amber-500/40 text-amber-400"
                : "border-border/40 text-muted-foreground hover:text-foreground"
            }`}
          >
            Semua Dokumen
          </button>
          {availableDocs.map((doc) => {
            const docId = doc.source_document_id || doc.doc_id;
            const shortName = doc.short_name || doc.label?.split(' tentang ')[0] || docId;
            const isActive = selectedDocId === docId;
            return (
              <button
                key={docId}
                onClick={() => setSelectedDocId(docId)}
                className={`text-xs px-3 py-1 rounded-full border transition-all flex items-center gap-1.5 ${
                  isActive
                    ? "bg-amber-500/15 border-amber-500/40 text-amber-400"
                    : "border-border/40 text-muted-foreground hover:text-foreground"
                }`}
              >
                <span
                  className="w-2 h-2 rounded-full"
                  style={{ backgroundColor: DOC_COLORS[docId] || "#888" }}
                />
                {shortName}
              </button>
            );
          })}
        </div>
      )}

      {/* Overview cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <Card className="bg-card/50 border-border/40">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground flex items-center gap-2">
              <Database className="h-4 w-4 text-amber-500" /> Total Nodes
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold">{stats.total_nodes}</p>
          </CardContent>
        </Card>
        <Card className="bg-card/50 border-border/40">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground flex items-center gap-2">
              <GitBranch className="h-4 w-4 text-blue-500" /> Total Edges
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold">{stats.total_edges}</p>
          </CardContent>
        </Card>
        <Card className="bg-card/50 border-border/40">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground flex items-center gap-2">
              <Layers className="h-4 w-4 text-green-500" /> Node Types
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold">{stats.node_types.length}</p>
          </CardContent>
        </Card>
        <Card className="bg-card/50 border-border/40">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-purple-500" /> Relation Types
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold">{stats.edge_types.length}</p>
          </CardContent>
        </Card>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {/* Node Distribution Bar Chart */}
        <Card className="bg-card/50 border-border/40">
          <CardHeader>
            <CardTitle className="text-base">Distribusi Node</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={stats.node_types} layout="vertical" margin={{ left: 20 }}>
                <XAxis type="number" stroke="#666" fontSize={11} />
                <YAxis
                  dataKey="label"
                  type="category"
                  width={120}
                  stroke="#666"
                  fontSize={11}
                  tick={// eslint-disable-next-line @typescript-eslint/no-explicit-any
                  ((props: any) => (
                    <text x={props.x} y={props.y} dy={4} textAnchor="end" fill={NODE_COLORS[props.payload?.value] || "#888"} fontSize={11}>
                      {props.payload?.value}
                    </text>
                  )) as any}
                />
                <Tooltip
                  contentStyle={{ backgroundColor: "#1a1a2e", border: "1px solid #333", borderRadius: "8px" }}
                  labelStyle={{ color: "#fff" }}
                />
                <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                  {stats.node_types.map((entry: TypeCount, i: number) => (
                    <Cell key={i} fill={NODE_COLORS[entry.label] || CHART_COLORS[i % CHART_COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Relation Types Pie Chart */}
        <Card className="bg-card/50 border-border/40">
          <CardHeader>
            <CardTitle className="text-base">Distribusi Relasi</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={stats.edge_types}
                  dataKey="count"
                  nameKey="label"
                  cx="50%"
                  cy="50%"
                  outerRadius={100}
                  innerRadius={50}
                  paddingAngle={2}
                  // eslint-disable-next-line @typescript-eslint/no-explicit-any
                  label={((props: any) => `${props.label} (${props.count || props.value})`) as any}
                  labelLine={false}
                  fontSize={10}
                >
                  {stats.edge_types.map((_: TypeCount, i: number) => (
                    <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ backgroundColor: "#1a1a2e", border: "1px solid #333", borderRadius: "8px" }}
                />
                <Legend fontSize={11} />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Tables */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="bg-card/50 border-border/40">
          <CardHeader>
            <CardTitle className="text-base">Detail Node Types</CardTitle>
          </CardHeader>
          <CardContent>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border/40">
                  <th className="text-left py-2 text-muted-foreground font-normal">Type</th>
                  <th className="text-right py-2 text-muted-foreground font-normal">Count</th>
                  <th className="text-right py-2 text-muted-foreground font-normal">%</th>
                </tr>
              </thead>
              <tbody>
                {stats.node_types.map((t: TypeCount) => (
                  <tr key={t.label} className="border-b border-border/20">
                    <td className="py-2 flex items-center gap-2">
                      <span
                        className="w-2.5 h-2.5 rounded-full"
                        style={{ backgroundColor: NODE_COLORS[t.label] || "#888" }}
                      />
                      {t.label}
                    </td>
                    <td className="text-right py-2 font-medium">{t.count}</td>
                    <td className="text-right py-2 text-muted-foreground">
                      {((t.count / stats.total_nodes) * 100).toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>

        <Card className="bg-card/50 border-border/40">
          <CardHeader>
            <CardTitle className="text-base">Detail Relation Types</CardTitle>
          </CardHeader>
          <CardContent>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border/40">
                  <th className="text-left py-2 text-muted-foreground font-normal">Type</th>
                  <th className="text-right py-2 text-muted-foreground font-normal">Count</th>
                  <th className="text-right py-2 text-muted-foreground font-normal">%</th>
                </tr>
              </thead>
              <tbody>
                {stats.edge_types.map((t: TypeCount, i: number) => (
                  <tr key={t.label} className="border-b border-border/20">
                    <td className="py-2 flex items-center gap-2">
                      <Badge
                        variant="outline"
                        className="text-[10px] px-1.5"
                        style={{ borderColor: CHART_COLORS[i % CHART_COLORS.length] }}
                      >
                        {t.label}
                      </Badge>
                    </td>
                    <td className="text-right py-2 font-medium">{t.count}</td>
                    <td className="text-right py-2 text-muted-foreground">
                      {((t.count / stats.total_edges) * 100).toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
