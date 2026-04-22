/**
 * LoraHeatmap — layer x rank heatmap by L2 norm.
 *
 * Cell color: L2 norm interpolated cyan->magenta.
 * Updates on metrics.lora_norms (every 50 steps).
 * Hover shows layer name + exact norm.
 */
import React, { useState, useMemo, useRef } from "react";
import { useStore } from "../lib/store";

interface CellTooltip {
  x: number;
  y: number;
  layerName: string;
  rankIdx: number;
  norm: number;
}

// Interpolate cyan->magenta based on normalized value 0-1
function normColor(t: number): string {
  // Cyan: #00e5ff, Magenta: #ff2bd6
  const r = Math.round(0 + t * 255);
  const g = Math.round(229 + t * (43 - 229));
  const b = Math.round(255 + t * (214 - 255));
  return `rgb(${r},${g},${b})`;
}

export function LoraHeatmap() {
  const metricsHistory = useStore((s) => s.metricsHistory);
  const activeRunId = useStore((s) => s.activeRunId);
  const [tooltip, setTooltip] = useState<CellTooltip | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Get the latest lora_norms
  const latestNorms = useMemo(() => {
    if (!activeRunId) return null;
    const history = metricsHistory[activeRunId] || [];
    // Find last entry with lora_norms
    for (let i = history.length - 1; i >= 0; i--) {
      if (history[i].lora_norms) return history[i].lora_norms!;
    }
    return null;
  }, [metricsHistory, activeRunId]);

  if (!latestNorms || Object.keys(latestNorms).length === 0) {
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
        NO LORA DATA
      </div>
    );
  }

  const layers = Object.keys(latestNorms);
  const maxRank = Math.max(...Object.values(latestNorms).map((v) => v.length));

  // Compute global min/max for color normalization
  const allValues = Object.values(latestNorms).flat();
  const minVal = Math.min(...allValues);
  const maxVal = Math.max(...allValues);
  const valRange = maxVal - minVal || 1;

  const cellSize = Math.max(6, Math.min(18, Math.floor(120 / layers.length)));
  const rankCellW = Math.max(4, Math.min(14, Math.floor(200 / maxRank)));

  return (
    <div ref={scrollRef} style={{ padding: "10px 14px", height: "100%", position: "relative", overflow: "auto" }}>
      {/* Header */}
      <div
        style={{
          fontSize: "9px",
          fontFamily: "'Share Tech Mono', monospace",
          letterSpacing: "2px",
          textTransform: "uppercase",
          color: "var(--dim)",
          marginBottom: 8,
        }}
      >
        LoRA Norms
        <span style={{ marginLeft: 8, color: "var(--cyan)" }}>{layers.length} layers</span>
      </div>

      {/* Color scale legend */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 6,
          marginBottom: 8,
        }}
      >
        <span style={{ fontSize: "9px", color: "var(--dim)" }}>
          {minVal.toFixed(3)}
        </span>
        <div
          style={{
            flex: 1,
            maxWidth: 80,
            height: 6,
            background: "linear-gradient(to right, var(--cyan), var(--magenta))",
          }}
        />
        <span style={{ fontSize: "9px", color: "var(--dim)" }}>
          {maxVal.toFixed(3)}
        </span>
      </div>

      {/* Heatmap grid */}
      <div
        style={{ display: "flex", flexDirection: "column", gap: 1 }}
        onMouseLeave={() => setTooltip(null)}
      >
        {layers.map((layerName, li) => {
          const norms = latestNorms[layerName];
          return (
            <div key={layerName} style={{ display: "flex", gap: 1, alignItems: "center" }}>
              {/* Layer label */}
              <div
                style={{
                  width: 80,
                  fontSize: "8px",
                  fontFamily: "'JetBrains Mono', monospace",
                  color: "var(--dim)",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                  flexShrink: 0,
                  textAlign: "right",
                  paddingRight: 4,
                }}
                title={layerName}
              >
                {layerName.replace(/^model\.(layers\.\d+\.)/, "L").slice(0, 12)}
              </div>
              {/* Rank cells */}
              {Array.from({ length: maxRank }, (_, ri) => {
                const val = norms[ri];
                if (val === undefined) {
                  return (
                    <div
                      key={ri}
                      style={{
                        width: rankCellW,
                        height: cellSize,
                        background: "var(--grid)",
                        opacity: 0.3,
                      }}
                    />
                  );
                }
                const t = (val - minVal) / valRange;
                const color = normColor(t);
                return (
                  <div
                    key={ri}
                    style={{
                      width: rankCellW,
                      height: cellSize,
                      background: color,
                      opacity: 0.85,
                      cursor: "default",
                    }}
                    onMouseEnter={(e) => {
                      const container = scrollRef.current;
                      if (!container) return;
                      const rect = e.currentTarget.getBoundingClientRect();
                      const parentRect = container.getBoundingClientRect();
                      setTooltip({
                        x: rect.left - parentRect.left + rankCellW,
                        y: rect.top - parentRect.top + container.scrollTop,
                        layerName,
                        rankIdx: ri,
                        norm: val,
                      });
                    }}
                  />
                );
              })}
            </div>
          );
        })}
      </div>

      {/* Rank axis */}
      <div
        style={{
          display: "flex",
          paddingLeft: 84,
          marginTop: 4,
          gap: 1,
        }}
      >
        {Array.from({ length: maxRank }, (_, i) => (
          <div
            key={i}
            style={{
              width: rankCellW,
              fontSize: "8px",
              textAlign: "center",
              color: "var(--dim)",
              fontFamily: "'JetBrains Mono', monospace",
            }}
          >
            {i % Math.max(1, Math.floor(maxRank / 8)) === 0 ? i : ""}
          </div>
        ))}
      </div>

      {/* Tooltip */}
      {tooltip && (
        <div
          style={{
            position: "absolute",
            left: tooltip.x + 8,
            top: tooltip.y,
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
          <div style={{ color: "var(--dim)", fontSize: "9px" }}>{tooltip.layerName}</div>
          <div>
            <span style={{ color: "var(--dim)" }}>rank </span>
            <span style={{ color: "var(--amber)" }}>{tooltip.rankIdx}</span>
          </div>
          <div>
            <span style={{ color: "var(--dim)" }}>norm </span>
            <span style={{ color: "var(--cyan)" }}>{tooltip.norm.toFixed(4)}</span>
          </div>
        </div>
      )}
    </div>
  );
}
