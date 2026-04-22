/**
 * Throughput — tokens/sec, steps/sec, H2D bandwidth.
 *
 * Three mini-cards, each with:
 * - Large tabular number (JetBrains Mono, 32px)
 * - Tiny inline sparkline (last 60 points)
 * - Color shifts lime->amber->magenta on coefficient-of-variation threshold
 */
import React, { useMemo } from "react";
import { useStore } from "../lib/store";
import { fmtTokPerSec, fmtNum, fmtGbps, cvColor } from "../lib/format";

function MiniSparkline({
  values,
  color,
  width = 80,
  height = 28,
}: {
  values: number[];
  color: string;
  width?: number;
  height?: number;
}) {
  if (values.length < 2) return <svg width={width} height={height} />;

  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;

  const pts = values
    .map((v, i) => {
      const x = (i / (values.length - 1)) * width;
      const y = height - ((v - min) / range) * (height - 2) - 1;
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <svg width={width} height={height} style={{ display: "block", overflow: "visible" }}>
      <polyline
        points={pts}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        opacity="0.8"
      />
    </svg>
  );
}

function MetricCard({
  label,
  value,
  unit,
  values,
}: {
  label: string;
  value: string;
  unit?: string;
  values: number[];
}) {
  const color = cvColor(values);

  return (
    <div
      style={{
        flex: 1,
        background: "var(--panel-hi)",
        padding: "12px 14px",
        display: "flex",
        flexDirection: "column",
        gap: 6,
      }}
    >
      <div
        style={{
          fontSize: "9px",
          fontFamily: "'Share Tech Mono', monospace",
          letterSpacing: "2px",
          textTransform: "uppercase",
          color: "var(--dim)",
        }}
      >
        {label}
      </div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 4 }}>
        <span
          className="num"
          style={{
            fontSize: "28px",
            fontFamily: "'JetBrains Mono', monospace",
            fontVariantNumeric: "tabular-nums",
            color,
            lineHeight: 1,
            transition: "color 0.3s ease",
          }}
        >
          {value}
        </span>
        {unit && (
          <span
            style={{
              fontSize: "10px",
              fontFamily: "'Share Tech Mono', monospace",
              color: "var(--dim)",
            }}
          >
            {unit}
          </span>
        )}
      </div>
      <MiniSparkline values={values} color={color} />
    </div>
  );
}

export function Throughput() {
  const metricsHistory = useStore((s) => s.metricsHistory);
  const activeRunId = useStore((s) => s.activeRunId);
  const streamStats = useStore((s) => s.streamStats);

  const data = useMemo(() => {
    if (!activeRunId) return [];
    return (metricsHistory[activeRunId] || []).slice(-60);
  }, [metricsHistory, activeRunId]);

  const tokValues = data.map((d) => d.tokens_per_sec || 0).filter((v) => v > 0);
  const stepsValues = data.map((d) => d.steps_per_sec || 0).filter((v) => v > 0);
  const h2dValues =
    streamStats && streamStats.h2d_bandwidth_gbps > 0
      ? [streamStats.h2d_bandwidth_gbps]
      : [];

  const latestToks = tokValues.length > 0 ? tokValues[tokValues.length - 1] : 0;
  const latestSteps = stepsValues.length > 0 ? stepsValues[stepsValues.length - 1] : 0;
  const latestH2D = h2dValues.length > 0 ? h2dValues[h2dValues.length - 1] : 0;

  return (
    <div
      style={{
        display: "flex",
        gap: 1,
        height: "100%",
        background: "var(--grid)",
      }}
    >
      <MetricCard
        label="Tokens/Sec"
        value={latestToks > 0 ? fmtNum(latestToks, 0) : "--"}
        unit="tok/s"
        values={tokValues}
      />
      <MetricCard
        label="Steps/Sec"
        value={latestSteps > 0 ? fmtNum(latestSteps, 2) : "--"}
        unit="it/s"
        values={stepsValues}
      />
      <MetricCard
        label="H2D Bandwidth"
        value={latestH2D > 0 ? fmtGbps(latestH2D).replace(" GB/s", "") : "--"}
        unit="GB/s"
        values={h2dValues}
      />
    </div>
  );
}
