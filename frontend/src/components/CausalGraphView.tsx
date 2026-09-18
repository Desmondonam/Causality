"use client";

import { useMemo } from "react";
import type { CausalEdge } from "@/lib/api";

function layoutGraph(nodes: string[], edges: CausalEdge[]) {
  const parents = new Map<string, string[]>();
  nodes.forEach((n) => parents.set(n, []));
  edges.forEach((e) => parents.get(e.target)?.push(e.source));

  // Longest-path layering (topological), so every edge points strictly
  // left-to-right - mirrors ml.causal_graph.draw_causal_graph in Python.
  const layer = new Map<string, number>();
  const resolve = (node: string, seen: Set<string> = new Set()): number => {
    if (layer.has(node)) return layer.get(node)!;
    if (seen.has(node)) return 0; // guard against unexpected cycles
    seen.add(node);
    const ps = parents.get(node) ?? [];
    const depth = ps.length === 0 ? 0 : Math.max(...ps.map((p) => resolve(p, seen) + 1));
    layer.set(node, depth);
    return depth;
  };
  nodes.forEach((n) => resolve(n));

  const byLayer = new Map<number, string[]>();
  nodes.forEach((n) => {
    const l = layer.get(n)!;
    byLayer.set(l, [...(byLayer.get(l) ?? []), n]);
  });

  const maxLayer = Math.max(...Array.from(byLayer.keys()));
  const colWidth = 190;
  const rowHeight = 64;
  const pos = new Map<string, { x: number; y: number }>();

  byLayer.forEach((layerNodes, l) => {
    const sorted = [...layerNodes].sort();
    sorted.forEach((n, i) => {
      const y = (i - (sorted.length - 1) / 2) * rowHeight;
      pos.set(n, { x: l * colWidth, y });
    });
  });

  const width = (maxLayer + 1) * colWidth;
  const maxRows = Math.max(...Array.from(byLayer.values()).map((l) => l.length));
  const height = maxRows * rowHeight + rowHeight;

  return { pos, width, height };
}

export function CausalGraphView({ nodes, edges }: { nodes: string[]; edges: CausalEdge[] }) {
  const { pos, width, height } = useMemo(() => layoutGraph(nodes, edges), [nodes, edges]);

  const padding = 90;
  const viewBox = `${-padding} ${-height / 2 - 20} ${width + padding * 2} ${height + 40}`;

  return (
    <div className="overflow-x-auto rounded-2xl border border-border bg-surface p-4">
      <svg viewBox={viewBox} className="mx-auto min-w-[720px]" style={{ height: Math.max(360, height + 40) }}>
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--muted)" />
          </marker>
        </defs>
        {edges.map((e, i) => {
          const s = pos.get(e.source);
          const t = pos.get(e.target);
          if (!s || !t) return null;
          return (
            <line
              key={i}
              x1={s.x + 44}
              y1={s.y}
              x2={t.x - 50}
              y2={t.y}
              stroke="var(--muted)"
              strokeWidth={1.5}
              markerEnd="url(#arrow)"
              opacity={0.7}
            />
          );
        })}
        {nodes.map((n) => {
          const p = pos.get(n);
          if (!p) return null;
          const isTarget = n === "diagnosis";
          const parts = n.split(" ");
          return (
            <g key={n} transform={`translate(${p.x}, ${p.y})`}>
              <rect
                x={-44}
                y={-16}
                width={88}
                height={32}
                rx={16}
                fill={isTarget ? "var(--malignant)" : "var(--accent)"}
              />
              <text textAnchor="middle" fontSize="9.5" fill="white" fontWeight={600}>
                {parts.length > 1 ? (
                  parts.map((part, i) => (
                    <tspan key={i} x={0} dy={i === 0 ? 4 - ((parts.length - 1) * 5) : 10}>
                      {part}
                    </tspan>
                  ))
                ) : (
                  <tspan dy={4}>{n}</tspan>
                )}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
