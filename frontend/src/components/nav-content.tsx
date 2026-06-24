"use client";

import Link from "next/link";
import { Network, MessageSquare, BarChart3, FileText, Home, Sun, Moon } from "lucide-react";
import { useTheme } from "@/lib/theme-context";

const navItems = [
  { href: "/", label: "Home", icon: Home },
  { href: "/explore", label: "Explorer", icon: Network },
  { href: "/qa", label: "QA", icon: MessageSquare },
  { href: "/analytics", label: "Analytics", icon: BarChart3 },
  { href: "/document", label: "Documents", icon: FileText },
];

export function NavContent() {
  const { theme, toggleTheme } = useTheme();

  return (
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
        {/* Theme Toggle */}
        <button
          onClick={toggleTheme}
          className="ml-2 p-2 rounded-md text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
          aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
          title={theme === "dark" ? "Light Mode" : "Dark Mode"}
        >
          {theme === "dark" ? (
            <Sun className="h-4 w-4" />
          ) : (
            <Moon className="h-4 w-4" />
          )}
        </button>
      </div>
    </div>
  );
}
