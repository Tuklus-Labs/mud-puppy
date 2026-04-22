/**
 * VramBar — horizontal stacked bar with VRAM segments.
 *
 * Segments: model / optimizer / activations / kv-cache / free / fragmentation
 * Hover shows exact GB + full torch.cuda.memory_stats detail panel.
 */
import React, { useState } from "react";
import { useStore } from "../lib/store";
import { fmtNum } from "../lib/format";

interface Segment {
  key: string;
  label: string;
  value: number; // GB
  color: string;
}

interface TooltipState {
  x: number;
  y: number;
  segment: Segment;
}

function computeSegments(
  memStats: {
    allocated_gb: number;
    reserved_gb: number;
    active_gb: number;
    fragmentation: number;
    model_gb?: number;
    optimizer_gb?: number;
    activations_gb?: number;
    kv_gb?: number;
  } | null,
  gpuHistory: { vram_used_gb: number; vram_total_gb: number }[]
): { segments: Segment[]; total: number } {
  if (!memStats) {
    // Fall back to GPU telemetry
    const latest = gpuHistory.length > 0 ? gpuHistory[gpuHistory.length - 1] : null;
    if (!latest) return { segments: [], total: 24 };
    return {
      segments: [
        {
          key: "used",
          label: "Used",
          value: latest.vram_used_gb,
          color: "var(--cyan)",
        },
        {
          key: "free",
          label: "Free",
          value: latest.vram_total_gb - latest.vram_used_gb,
          color: "var(--grid)",
        },
      ],
      total: latest.vram_total_gb,
    };
  }

  const model = memStats.model_gb || 0;
  const optimizer = memStats.optimizer_gb || 0;
  const activations = memStats.activations_gb || 0;
  const kv = memStats.kv_gb || 0;

  // Unknown "other" allocated
  const accounted = model + optimizer + activations + kv;
  const other = Math.max(memStats.allocated_gb - accounted, 0);
  const frag = memStats.reserved_gb * memStats.fragmentation;
  const free = Math.max(memStats.reserved_gb - memStats.allocated_gb - frag, 0);

  const segments: Segment[] = [
    { key: "model", label: "Model", value: model, color: "var(--cyan)" },
    { key: "optimizer", label: "Optimizer", value: optimizer, color: "var(--amber)" },
    { key: "activations", label: "Activations", value: activations, color: "var(--lime)" },
    { key: "kv", label: "KV Cache", value: kv, color: "#7c5cff" },
    { key: "other", label: "Other", value: other, color: "var(--dim)" },
    { key: "frag", label: "Fragmentation", value: frag, color: "var(--magenta)" },
    { key: "free", label: "Free", value: free, color: "var(--grid)" },
  ].filter((s) => s.value > 0.001);

  return { segments, total: memStats.reserved_gb };
}

export function VramBar() {
  const memStats = useStore((s) => s.memoryStats);
  const gpuHistory = useStore((s) => s.gpuHistory);
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);

  const { segments, total } = computeSegments(memStats, gpuHistory);
  const totalLabeled = segments.reduce((s, seg) => s + seg.value, 0);
  const displayTotal = Math.max(total, totalLabeled, 1);

  return (
    <div style={{ padding: "10px 14px", height: "100%" }}>
      {/* Title row */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginBottom: 8,
          alignItems: "baseline",
        }}
      >
        <span
          style={{
            fontSize: "9px",
            fontFamily: "'Share Tech Mono', monospace",
            letterSpacing: "2px",
            textTransform: "uppercase",
            color: "var(--dim)",
          }}
        >
          VRAM
        </span>
        <span
          className="num"
          style={{
            fontSize: "11px",
            fontFamily: "'JetBrains Mono', monospace",
            color: "var(--text)",
          }}
        >
          {fmtNum(totalLabeled - (segments.find((s) => s.key === "free")?.value || 0), 2)}
          <span style={{ color: "var(--dim)" }}>
            {" / "}{fmtNum(displayTotal, 1)} GB
          </span>
        </span>
      </div>

      {/* Stacked bar */}
      <div
        style={{
          display: "flex",
          height: 20,
          background: "var(--grid)",
          overflow: "hidden",
          position: "relative",
        }}
        onMouseLeave={() => setTooltip(null)}
      >
        {segments.map((seg) => {
          const pct = (seg.value / displayTotal) * 100;
          return (
            <div
              key={seg.key}
              title={`${seg.label}: ${fmtNum(seg.value, 2)} GB`}
              style={{
                width: `${pct}%`,
                background: seg.color,
                opacity: seg.key === "free" ? 0.15 : 0.85,
                transition: "width 0.3s ease",
                cursor: "default",
                position: "relative",
              }}
              onMouseEnter={(e) => {
                const parent = e.currentTarget.parentElement;
                if (!parent) return;
                const rect = e.currentTarget.getBoundingClientRect();
                const parentRect = parent.getBoundingClientRect();
                setTooltip({
                  x: rect.left - parentRect.left + rect.width / 2,
                  y: -8,
                  segment: seg,
                });
              }}
            />
          );
        })}
      </div>

      {/* Legend */}
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: "6px 14px",
          marginTop: 8,
        }}
      >
        {segments
          .filter((s) => s.key !== "free")
          .map((seg) => (
            <div
              key={seg.key}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 4,
                fontSize: "10px",
                color: "var(--dim)",
              }}
            >
              <div
                style={{
                  width: 8,
                  height: 8,
                  background: seg.color,
                  flexShrink: 0,
                }}
              />
              <span>{seg.label}</span>
              <span
                className="num"
                style={{
                  color: seg.color,
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: "10px",
                }}
              >
                {fmtNum(seg.value, 2)}G
              </span>
            </div>
          ))}
      </div>

      {/* Hover tooltip */}
      {tooltip && (
        <div
          style={{
            position: "absolute",
            left: tooltip.x,
            top: tooltip.y,
            transform: "translateX(-50%)",
            background: "var(--panel-hi)",
            border: "1px solid var(--border)",
            padding: "4px 10px",
            fontSize: "10px",
            fontFamily: "'JetBrains Mono', monospace",
            color: "var(--text)",
            pointerEvents: "none",
            whiteSpace: "nowrap",
            zIndex: 10,
          }}
        >
          {tooltip.segment.label}: {fmtNum(tooltip.segment.value, 3)} GB
          {memStats && (
            <>
              <div style={{ color: "var(--dim)", fontSize: "9px", marginTop: 3 }}>
                allocated: {fmtNum(memStats.allocated_gb, 2)} GB
              </div>
              <div style={{ color: "var(--dim)", fontSize: "9px" }}>
                reserved:  {fmtNum(memStats.reserved_gb, 2)} GB
              </div>
              <div style={{ color: "var(--dim)", fontSize: "9px" }}>
                frag: {(memStats.fragmentation * 100).toFixed(1)}%
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
