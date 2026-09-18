"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity } from "lucide-react";
import clsx from "clsx";

const LINKS = [
  { href: "/", label: "Home" },
  { href: "/predict", label: "Predict" },
  { href: "/insights", label: "Insights" },
  { href: "/methodology", label: "Methodology" },
];

export function Navbar() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-50 border-b border-border bg-surface/80 backdrop-blur-md">
      <nav className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3 sm:px-6">
        <Link href="/" className="flex items-center gap-2 font-semibold text-foreground">
          <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-brand text-brand-foreground">
            <Activity size={18} />
          </span>
          <span>Causality</span>
        </Link>
        <div className="flex items-center gap-1 sm:gap-2">
          {LINKS.map((link) => {
            const active = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                className={clsx(
                  "rounded-full px-3 py-1.5 text-sm font-medium transition-colors",
                  active
                    ? "bg-brand text-brand-foreground"
                    : "text-muted hover:bg-border/60 hover:text-foreground",
                )}
              >
                {link.label}
              </Link>
            );
          })}
        </div>
      </nav>
    </header>
  );
}
