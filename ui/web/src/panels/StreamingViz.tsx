/**
 * StreamingViz — vertical stack of transformer layer rectangles.
 *
 * GPU-resident layers glow cyan.
 * Currently-prefetching layer pulses amber.
 * CPU-resident layers dim.
 * Hover shows layer index + last H2D time.
 */
import React, { useState } from "react";
import { useStore } from "../lib/store";

interface LayerTooltip {
  idx: number;
  x: number;
  y: number;
  resident: boolean;
  prefetching: boolean;
  h2d_ms?: number;
}

export function StreamingViz() {
  const streamStats = useStore((s) => s.streamStats);
  const [tooltip, setTooltip] = useState<LayerTooltip | null>(null);

  if (!streamStats) {
    return (
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          height: "100%",
          color: "var(--dim)",
          fontFamily: "'Share Tech Mono', monospace",
          fontSize: "10px",
          letterSpacing: "2px",
        }}
      >
        STREAMING INACTIVE
      </div>
    );
  }

  const { layers_total, layer_states } = streamStats;
  const totalLayers = layers_total || 32;

  // Build synthetic layer states if detailed states not available
  const states = layer_states || Array.from({ length: totalLayers }, (_, i) => ({
    idx: i,
    resident: i < streamStats.layers_resident,
    prefetching: i === streamStats.layers_resident,
    last_h2d_ms: undefined,
  }));

  // Layout: fit all layers in available space
  const layerH = Math.max(4, Math.min(20, Math.floor(200 / totalLayers)));
  const layerGap = 1;

  return (
    <div style={{ padding: "10px 14px", height: "100%", position: "relative" }}>
      {/* Header */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginBottom: 8,
          fontSize: "9px",
          fontFamily: "'Share Tech Mono', monospace",
          letterSpacing: "2px",
          textTransform: "uppercase",
        }}
      >
        <span style={{ color: "var(--dim)" }}>Layers</span>
        <span style={{ color: "var(--cyan)" }}>
          {streamStats.layers_resident}/{totalLayers} resident
        </span>
      </div>

      {/* Stats strip */}
      <div
        style={{
          display: "flex",
          gap: 12,
          marginBottom: 10,
          fontSize: "10px",
          fontFamily: "'JetBrains Mono', monospace",
        }}
      >
        <span>
          <span style={{ color: "var(--dim)" }}>BW </span>
          <span style={{ color: "var(--lime)" }}>
            {streamStats.h2d_bandwidth_gbps.toFixed(1)} GB/s
          </span>
        </span>
        <span>
          <span style={{ color: "var(--dim)" }}>HIT </span>
          <span style={{ color: "var(--amber)" }}>
            {(streamStats.prefetch_hit_rate * 100).toFixed(0)}%
          </span>
        </span>
      </div>

      {/* Layer stack */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: layerGap,
          overflowY: "auto",
          maxHeight: "calc(100% - 60px)",
        }}
        onMouseLeave={() => setTooltip(null)}
      >
        {states.map((s) => {
          let bg = "var(--border)";
          let glow = "";
          let anim = "";

          if (s.prefetching) {
            bg = "var(--amber)";
            anim = "pulse-amber 0.8s ease-in-out infinite";
          } else if (s.resident) {
            bg = "var(--cyan)";
            glow = "0 0 4px rgba(0, 229, 255, 0.6)";
          }

          return (
            <div
              key={s.idx}
              style={{
                height: layerH,
                background: bg,
                boxShadow: glow || undefined,
                animation: anim || undefined,
                opacity: s.resident || s.prefetching ? 1 : 0.25,
                cursor: "default",
                transition: "background 0.15s, opacity 0.15s",
                position: "relative",
              }}
              onMouseEnter={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                const parentRect = e.currentTarget.parentElement!.getBoundingClientRect();
                setTooltip({
                  idx: s.idx,
                  x: rect.right - parentRect.left + 4,
                  y: rect.top - parentRect.top,
                  resident: s.resident,
                  prefetching: s.prefetching,
                  h2d_ms: s.last_h2d_ms,
                });
              }}
            >
              {/* Layer index label at regular intervals */}
              {(s.idx % Math.max(1, Math.floor(totalLayers / 8)) === 0 || s.idx === totalLayers - 1) && (
                <span
                  style={{
                    position: "absolute",
                    right: "calc(100% + 4px)",
                    top: "50%",
                    transform: "translateY(-50%)",
                    fontSize: "9px",
                    fontFamily: "'JetBrains Mono', monospace",
                    color: "var(--dim)",
                    whiteSpace: "nowrap",
                  }}
                >
                  {s.idx}
                </span>
              )}
            </div>
          );
        })}
      </div>

      {/* Tooltip */}
      {tooltip && (
        <div
          style={{
            position: "absolute",
            left: tooltip.x + 20,
            top: Math.min(tooltip.y, 150),
            background: "var(--panel-hi)",
            border: "1px solid var(--border)",
            padding: "4px 10px",
            fontSize: "10px",
            fontFamily: "'JetBrains Mono', monospace",
            color: "var(--text)",
            pointerEvents: "none",
            zIndex: 10,
            whiteSpace: "nowrap",
          }}
        >
          <div>
            <span style={{ color: "var(--dim)" }}>layer </span>
            <span style={{ color: "var(--amber)" }}>{tooltip.idx}</span>
          </div>
          <div>
            <span
              style={{
                color: tooltip.prefetching
                  ? "var(--amber)"
                  : tooltip.resident
                  ? "var(--cyan)"
                  : "var(--dim)",
              }}
            >
              {tooltip.prefetching ? "prefetching" : tooltip.resident ? "resident" : "CPU"}
            </span>
          </div>
          {tooltip.h2d_ms !== undefined && (
            <div>
              <span style={{ color: "var(--dim)" }}>H2D </span>
              <span style={{ color: "var(--lime)" }}>{tooltip.h2d_ms.toFixed(1)}ms</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
