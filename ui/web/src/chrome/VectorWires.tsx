/**
 * VectorWires — animated polylines between panel anchor elements.
 *
 * Thicken briefly on matching IPC events (stroke-width 1->3->1 over 400ms).
 * Opacity 0.4 -> 1 -> 0.4.
 *
 * Active state lives in a ref (not useState) so the recalculate callback
 * never closes over a stale snapshot. Rendering is triggered via a plain
 * forceUpdate tick when activity changes.
 */
import React, { useEffect, useRef, useState, useCallback } from "react";
import { useStore } from "../lib/store";

interface Wire {
  from: string;  // element id
  to: string;    // element id
  event?: string;
}

interface WireGeom {
  key: string;
  points: string;
}

interface VectorWiresProps {
  wires: Wire[];
}

function getCenter(id: string, svgRect: DOMRect): { x: number; y: number } | null {
  const el = document.getElementById(id);
  if (!el) return null;
  const r = el.getBoundingClientRect();
  return {
    x: r.left + r.width / 2 - svgRect.left,
    y: r.top + r.height / 2 - svgRect.top,
  };
}

function buildPolylinePoints(
  from: { x: number; y: number },
  to: { x: number; y: number }
): string {
  const mx = (from.x + to.x) / 2;
  return `${from.x},${from.y} ${mx},${from.y} ${mx},${to.y} ${to.x},${to.y}`;
}

export function VectorWires({ wires }: VectorWiresProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const lastEvent = useStore((s) => s.lastEvent);

  // Geometry (points per wire) lives in ref + state for render.
  const [geom, setGeom] = useState<WireGeom[]>([]);

  // Active-state ref: source of truth; never a stale closure.
  const activeRef = useRef<Map<string, boolean>>(new Map());
  const timersRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());

  // Tick counter used solely to force a render when active flags change.
  const [, setTick] = useState(0);
  const bumpTick = useCallback(() => setTick((t) => (t + 1) & 0xffff), []);

  const recalculate = useCallback(() => {
    if (!svgRef.current) return;
    const svgRect = svgRef.current.getBoundingClientRect();
    const next: WireGeom[] = wires.map((w, i) => {
      const from = getCenter(w.from, svgRect);
      const to = getCenter(w.to, svgRect);
      const key = `wire-${i}`;
      if (!from || !to) return { key, points: "" };
      return { key, points: buildPolylinePoints(from, to) };
    });
    setGeom(next);
  }, [wires]);

  // Recalculate on resize + mount
  useEffect(() => {
    recalculate();
    const ro = new ResizeObserver(recalculate);
    const parent = svgRef.current?.parentElement;
    if (parent) ro.observe(parent);
    return () => ro.disconnect();
  }, [recalculate]);

  // Flash wires on matching events. Writes into activeRef directly.
  useEffect(() => {
    if (!lastEvent) return;
    wires.forEach((w, i) => {
      if (w.event && w.event === lastEvent) {
        const key = `wire-${i}`;
        const prev = timersRef.current.get(key);
        if (prev) clearTimeout(prev);

        activeRef.current.set(key, true);
        bumpTick();

        const t = setTimeout(() => {
          activeRef.current.set(key, false);
          timersRef.current.delete(key);
          bumpTick();
        }, 400);
        timersRef.current.set(key, t);
      }
    });
  }, [lastEvent, wires, bumpTick]);

  // I12: clear all pending timers on unmount.
  useEffect(() => {
    const timers = timersRef.current;
    return () => {
      timers.forEach((t) => clearTimeout(t));
      timers.clear();
    };
  }, []);

  return (
    <svg
      ref={svgRef}
      style={{
        position: "absolute",
        inset: 0,
        width: "100%",
        height: "100%",
        pointerEvents: "none",
        zIndex: 5,
        overflow: "visible",
      }}
      aria-hidden="true"
    >
      <defs>
        <filter id="wire-glow">
          <feGaussianBlur stdDeviation="1.5" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
      {geom.map((g) => {
        if (!g.points) return null;
        const active = activeRef.current.get(g.key) === true;
        return (
          <polyline
            key={g.key}
            points={g.points}
            fill="none"
            stroke="var(--cyan)"
            strokeWidth={active ? 2.5 : 1}
            opacity={active ? 0.9 : 0.35}
            strokeDasharray={active ? undefined : "4,6"}
            filter={active ? "url(#wire-glow)" : undefined}
            style={{
              transition: "stroke-width 0.15s ease, opacity 0.15s ease",
            }}
          />
        );
      })}
    </svg>
  );
}
