import type { Metadata } from "next";
import { Geist } from "next/font/google";
import "./globals.css";
import { ThemeProvider } from "@/lib/theme-context";
import { NavContent } from "@/components/nav-content";

const geist = Geist({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Legal KG Visualization",
  description: "LLM-Driven Indonesian Legal Knowledge Graph Visualization",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="id" suppressHydrationWarning>
      <body className={`${geist.className} antialiased bg-background text-foreground min-h-screen`}>
        <ThemeProvider>
          {/* Navbar */}
          <nav className="sticky top-0 z-50 border-b border-border/40 bg-background/80 backdrop-blur-lg">
            <NavContent />
          </nav>
          <main>{children}</main>
        </ThemeProvider>
      </body>
    </html>
  );
}
