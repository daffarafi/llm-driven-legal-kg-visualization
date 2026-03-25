import type { Metadata } from "next";
import { Geist } from "next/font/google";
import Link from "next/link";
import "./globals.css";
import { Network, MessageSquare, BarChart3, FileText, Home } from "lucide-react";

const geist = Geist({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Legal KG Visualization",
  description: "LLM-Driven Indonesian Legal Knowledge Graph Visualization",
};

const navItems = [
  { href: "/", label: "Home", icon: Home },
  { href: "/explore", label: "Explorer", icon: Network },
  { href: "/qa", label: "QA", icon: MessageSquare },
  { href: "/analytics", label: "Analytics", icon: BarChart3 },
  { href: "/document", label: "Documents", icon: FileText },
];

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="id" className="dark">
      <body className={`${geist.className} antialiased bg-background text-foreground min-h-screen`}>
        {/* Navbar */}
        <nav className="sticky top-0 z-50 border-b border-border/40 bg-background/80 backdrop-blur-lg">
          <div className="max-w-[1400px] mx-auto flex items-center justify-between h-14 px-4">
            <Link href="/" className="flex items-center gap-2 font-bold text-lg">
              <Network className="h-5 w-5 text-amber-500" />
              <span className="bg-gradient-to-r from-amber-500 to-orange-500 bg-clip-text text-transparent">
                Legal KG
              </span>
            </Link>
            <div className="flex items-center gap-1">
              {navItems.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-sm text-muted-foreground hover:text-foreground rounded-md hover:bg-accent transition-colors"
                >
                  <item.icon className="h-4 w-4" />
                  {item.label}
                </Link>
              ))}
            </div>
          </div>
        </nav>
        <main>{children}</main>
      </body>
    </html>
  );
}
