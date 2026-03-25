"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { FileText, ChevronRight, BookOpen } from "lucide-react";
import { getDocuments } from "@/lib/api";

interface DocSummary {
  id: string;
  label: string;
  bab_count: number;
  pasal_count: number;
}

export default function DocumentsListPage() {
  const [docs, setDocs] = useState<DocSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getDocuments()
      .then((data) => setDocs((data as { documents: DocSummary[] }).documents || []))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="max-w-[800px] mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold mb-2 flex items-center gap-2">
        <BookOpen className="h-6 w-6 text-amber-500" /> Dokumen Hukum
      </h1>
      <p className="text-sm text-muted-foreground mb-6">
        Pilih dokumen untuk membaca teks lengkap dengan navigasi KG.
      </p>

      {loading ? (
        <p className="text-muted-foreground">Loading...</p>
      ) : docs.length === 0 ? (
        <p className="text-muted-foreground">Tidak ada dokumen. Pastikan Neo4j berjalan.</p>
      ) : (
        <div className="space-y-3">
          {docs.map((doc) => (
            <Link key={doc.id} href={`/document/${encodeURIComponent(doc.id)}`}>
              <Card className="bg-card/50 border-border/40 hover:border-amber-500/40 transition-all cursor-pointer">
                <CardContent className="flex items-center justify-between py-4">
                  <div className="flex items-center gap-3">
                    <FileText className="h-8 w-8 text-amber-500/60" />
                    <div>
                      <p className="font-medium">{doc.label}</p>
                      <p className="text-xs text-muted-foreground">
                        {doc.bab_count} Bab · {doc.pasal_count} Pasal
                      </p>
                    </div>
                  </div>
                  <ChevronRight className="h-4 w-4 text-muted-foreground" />
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
