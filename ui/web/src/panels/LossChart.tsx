/**
 * LossChart — D3 SVG line chart for training loss.
 *
 * Features:
 * - Ring-decimated to 1k points for rendering performance
 * - Log-Y toggle
 * - Step / wall-clock X toggle
 * - Hover crosshair with tooltip
 * - Two-run overlay (compare mode)
 */
import React, { useRef, useEffect, useState, useCallback } from "react";
import * as d3 from "d3";
import { useStore } from "../lib/store";
import type { MetricsEvent } from "../lib/ipc-types";
import { fmtLoss } from "../lib/format";

const MARGIN = { top: 12, right: 24, bottom: 32, left: 52 };
const MAX_RENDER_POINTS = 1000;

function decimate(data: MetricsEvent[], max: number): MetricsEvent[] {
  if (data.length <= max) return data;
  const step = Math.ceil(data.length / max);
  return data.filter((_, i) => i % step === 0);
}

interface CrosshairState {
  x: number;
  y: number;
  step: number;
  loss: number;
  run_id: string;
}

interface LossChartProps {
  runId: string;
}

export function LossChart({ runId }: LossChartProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [logY, setLogY] = useState(false);
  const [xMode, setXMode] = useState<"step" | "wall">("step");
  const [crosshair, setCrosshair] = useState<CrosshairState | null>(null);
  const [size, setSize] = useState({ width: 400, height: 200 });

  const metricsHistory = useStore((s) => s.metricsHistory);
  const compareRunId = useStore((s) => s.compareRunId);

  const primaryData = decimate(metricsHistory[runId] || [], MAX_RENDER_POINTS);
  const compareData = compareRunId
    ? decimate(metricsHistory[compareRunId] || [], MAX_RENDER_POINTS)
    : [];

  // Track container size
  useEffect(() => {
    if (!svgRef.current) return;
    const ro = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) {
        setSize({
          width: entry.contentRect.width,
          height: entry.contentRect.height,
        });
      }
    });
    ro.observe(svgRef.current.parentElement!);
    return () => ro.disconnect();
  }, []);

  const draw = useCallback(() => {
    if (!svgRef.current || primaryData.length < 2) return;

    const { width, height } = size;
    const innerW = width - MARGIN.left - MARGIN.right;
    const innerH = height - MARGIN.top - MARGIN.bottom;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const xAccessor = xMode === "step" ? (d: MetricsEvent) => d.step : (_d: MetricsEvent, i: number) => i;
    const allData = [...primaryData, ...compareData];
    const xExtent = d3.extent(allData.map(xAccessor)) as [number, number];
    const allLoss = allData.map((d) => d.loss).filter((v) => v > 0 && isFinite(v));
    const yExtent = d3.extent(allLoss) as [number, number];

    const xScale = d3.scaleLinear().domain(xExtent).range([0, innerW]).nice();

    const yScale = logY
      ? d3
          .scaleLog()
          .domain([Math.max(yExtent[0] * 0.9, 1e-6), yExtent[1] * 1.1])
          .range([innerH, 0])
          .nice()
      : d3
          .scaleLinear()
          .domain([Math.max(yExtent[0] * 0.9, 0), yExtent[1] * 1.1])
          .range([innerH, 0])
          .nice();

    const g = svg
      .append("g")
      .attr("transform", `translate(${MARGIN.left},${MARGIN.top})`);

    // Grid lines
    g.append("g")
      .attr("class", "grid-y")
      .call(
        d3
          .axisLeft(yScale)
          .ticks(5)
          .tickSize(-innerW)
          .tickFormat(() => "")
      )
      .call((gEl) => {
        gEl.select(".domain").remove();
        gEl.selectAll("line")
          .attr("stroke", "var(--grid)")
          .attr("stroke-dasharray", "3,4")
          .attr("opacity", "0.6");
      });

    // X axis
    g.append("g")
      .attr("transform", `translate(0,${innerH})`)
      .call(d3.axisBottom(xScale).ticks(6))
      .call((gEl) => {
        gEl.select(".domain").attr("stroke", "var(--border)");
        gEl.selectAll("text")
          .attr("fill", "var(--dim)")
          .attr("font-family", "'JetBrains Mono', monospace")
          .attr("font-size", "10");
        gEl.selectAll("line").attr("stroke", "var(--border)");
      });

    // Y axis
    g.append("g")
      .call(logY ? d3.axisLeft(yScale).ticks(5, ".2~e") : d3.axisLeft(yScale).ticks(5))
      .call((gEl) => {
        gEl.select(".domain").attr("stroke", "var(--border)");
        gEl.selectAll("text")
          .attr("fill", "var(--dim)")
          .attr("font-family", "'JetBrains Mono', monospace")
          .attr("font-size", "10");
        gEl.selectAll("line").attr("stroke", "var(--border)");
      });

    // Line generator
    const line = d3
      .line<MetricsEvent>()
      .x((d, i) => xScale(xMode === "step" ? d.step : i))
      .y((d) => yScale(Math.max(d.loss, logY ? 1e-6 : 0)))
      .defined((d) => isFinite(d.loss) && d.loss > 0)
      .curve(d3.curveMonotoneX);

    // Primary line
    if (primaryData.length > 1) {
      g.append("path")
        .datum(primaryData)
        .attr("fill", "none")
        .attr("stroke", "var(--cyan)")
        .attr("stroke-width", 1.5)
        .attr("d", line);
    }

    // Compare line (magenta)
    if (compareData.length > 1) {
      g.append("path")
        .datum(compareData)
        .attr("fill", "none")
        .attr("stroke", "var(--magenta)")
        .attr("stroke-width", 1.5)
        .attr("d", line);
    }

    // Crosshair line (rendered after lines so it's on top)
    const vLine = g
      .append("line")
      .attr("stroke", "var(--amber)")
      .attr("stroke-width", 1)
      .attr("stroke-dasharray", "3,3")
      .attr("opacity", 0)
      .attr("y1", 0)
      .attr("y2", innerH);

    // Invisible overlay for mouse events
    g.append("rect")
      .attr("width", innerW)
      .attr("height", innerH)
      .attr("fill", "transparent")
      .on("mousemove", (event: MouseEvent) => {
        const [mx] = d3.pointer(event);
        const xVal = xScale.invert(mx);

        // Find nearest point and its index in one pass.
        let nearest: MetricsEvent | null = null;
        let nearestIdx = -1;
        let bestDist = Infinity;
        primaryData.forEach((d, i) => {
          const xd = xMode === "step" ? d.step : i;
          const dist = Math.abs(xd - xVal);
          if (dist < bestDist) {
            bestDist = dist;
            nearest = d;
            nearestIdx = i;
          }
        });

        if (nearest && nearestIdx >= 0) {
          const ne = nearest as MetricsEvent;
          const xPx = xScale(xMode === "step" ? ne.step : nearestIdx);
          vLine.attr("x1", xPx).attr("x2", xPx).attr("opacity", 0.8);
          setCrosshair({
            x: xPx + MARGIN.left,
            y: yScale(Math.max(ne.loss, logY ? 1e-6 : 0)) + MARGIN.top,
            step: ne.step,
            loss: ne.loss,
            run_id: ne.run_id,
          });
        }
      })
      .on("mouseleave", () => {
        vLine.attr("opacity", 0);
        setCrosshair(null);
      });

    // X axis label
    svg
      .append("text")
      .attr("x", MARGIN.left + innerW / 2)
      .attr("y", height - 4)
      .attr("text-anchor", "middle")
      .attr("fill", "var(--dim)")
      .attr("font-family", "'Share Tech Mono', monospace")
      .attr("font-size", "9")
      .attr("letter-spacing", "1")
      .text(xMode === "step" ? "STEP" : "SAMPLE");

    // Y axis label
    svg
      .append("text")
      .attr("transform", `rotate(-90)`)
      .attr("x", -(MARGIN.top + innerH / 2))
      .attr("y", 14)
      .attr("text-anchor", "middle")
      .attr("fill", "var(--dim)")
      .attr("font-family", "'Share Tech Mono', monospace")
      .attr("font-size", "9")
      .attr("letter-spacing", "1")
      .text("LOSS");
  }, [primaryData, compareData, logY, xMode, size]);

  useEffect(() => {
    draw();
  }, [draw]);

  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      {/* Controls */}
      <div
        style={{
          position: "absolute",
          top: 4,
          right: 8,
          display: "flex",
          gap: 6,
          zIndex: 2,
        }}
      >
        <button
          className={logY ? "active" : ""}
          onClick={() => setLogY((v) => !v)}
          style={{
            fontSize: "9px",
            padding: "2px 6px",
            borderColor: logY ? "var(--cyan)" : "var(--border)",
            color: logY ? "var(--cyan)" : "var(--dim)",
          }}
        >
          LOG-Y
        </button>
        <button
          onClick={() => setXMode((m) => (m === "step" ? "wall" : "step"))}
          style={{
            fontSize: "9px",
            padding: "2px 6px",
            borderColor: "var(--border)",
            color: "var(--dim)",
          }}
        >
          {xMode === "step" ? "STEP" : "WALL"}
        </button>
      </div>

      {/* SVG chart */}
      <svg
        ref={svgRef}
        style={{ width: "100%", height: "100%", display: "block" }}
      />

      {/* Crosshair tooltip */}
      {crosshair && (
        <div
          style={{
            position: "absolute",
            left: crosshair.x + 8,
            top: crosshair.y - 24,
            background: "var(--panel-hi)",
            border: "1px solid var(--border)",
            padding: "3px 8px",
            fontSize: "10px",
            fontFamily: "'JetBrains Mono', monospace",
            color: "var(--text)",
            pointerEvents: "none",
            whiteSpace: "nowrap",
            zIndex: 5,
          }}
        >
          <span style={{ color: "var(--dim)" }}>step </span>
          <span style={{ color: "var(--amber)" }}>{crosshair.step}</span>
          <span style={{ color: "var(--dim)" }}> loss </span>
          <span style={{ color: "var(--cyan)" }}>{fmtLoss(crosshair.loss)}</span>
        </div>
      )}

      {/* Empty state */}
      {primaryData.length < 2 && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "var(--dim)",
            fontFamily: "'Share Tech Mono', monospace",
            fontSize: "11px",
            letterSpacing: "2px",
            textTransform: "uppercase",
          }}
        >
          AWAITING METRICS
        </div>
      )}
    </div>
  );
}
