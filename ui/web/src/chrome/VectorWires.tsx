/**
 * VectorWires — animated polylines between panel anchor elements.
 *
 * Thicken briefly on matching IPC events (stroke-width 1->3->1 over 400ms).
 * Opacity 0.4 -> 1 -> 0.4.
 */
import React, { useEffect, useRef, useState, useCallback } from "react";
import { useStore } from "../lib/store";

interface WireAnchor {
  id: string;        // DOM element id for source/target
  event?: string;    // IPC event name that triggers this wire
}

interface Wire {
  from: string;  // element id
  to: string;    // element id
  event?: string;
}

interface WireState {
  key: string;
  points: string; // SVG polyline points
  active: boolean;
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
  // Elbow routing: horizontal then vertical midpoint
  const mx = (from.x + to.x) / 2;
  return `${from.x},${from.y} ${mx},${from.y} ${mx},${to.y} ${to.x},${to.y}`;
}

export function VectorWires({ wires }: VectorWiresProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [wireStates, setWireStates] = useState<WireState[]>([]);
  const lastEvent = useStore((s) => s.lastEvent);
  const activeTimers = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());

  const recalculate = useCallback(() => {
    if (!svgRef.current) return;
    const svgRect = svgRef.current.getBoundingClientRect();
    const states: WireState[] = wires.map((w, i) => {
      const from = getCenter(w.from, svgRect);
      const to = getCenter(w.to, svgRect);
      const key = `wire-${i}`;
      if (!from || !to) return { key, points: "", active: false };
      return {
        key,
        points: buildPolylinePoints(from, to),
        active: wireStates.find((s) => s.key === key)?.active ?? false,
      };
    });
    setWireStates(states);
  }, [wires]); // eslint-disable-line react-hooks/exhaustive-deps

  // Recalculate on resize
  useEffect(() => {
    recalculate();
    const ro = new ResizeObserver(recalculate);
    if (svgRef.current?.parentElement) {
      ro.observe(svgRef.current.parentElement);
    }
    return () => ro.disconnect();
  }, [recalculate]);

  // Flash wires on matching events
  useEffect(() => {
    if (!lastEvent) return;
    wires.forEach((w, i) => {
      if (w.event && w.event === lastEvent) {
        const key = `wire-${i}`;
        // Clear existing timer
        const existing = activeTimers.current.get(key);
        if (existing) clearTimeout(existing);

        // Activate
        setWireStates((prev) =>
          prev.map((s) => (s.key === key ? { ...s, active: true } : s))
        );

        // Deactivate after 400ms
        const t = setTimeout(() => {
          setWireStates((prev) =>
            prev.map((s) => (s.key === key ? { ...s, active: false } : s))
          );
          activeTimers.current.delete(key);
        }, 400);
        activeTimers.current.set(key, t);
      }
    });
  }, [lastEvent, wires]);

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
      {wireStates.map((ws) => {
        if (!ws.points) return null;
        return (
          <polyline
            key={ws.key}
            points={ws.points}
            fill="none"
            stroke="var(--cyan)"
            strokeWidth={ws.active ? 2.5 : 1}
            opacity={ws.active ? 0.9 : 0.35}
            strokeDasharray={ws.active ? undefined : "4,6"}
            filter={ws.active ? "url(#wire-glow)" : undefined}
            style={{
              transition: "stroke-width 0.15s ease, opacity 0.15s ease",
            }}
          />
        );
      })}
    </svg>
  );
}
