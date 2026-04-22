/**
 * Monitor pane — live training dashboard.
 *
 * Grid layout:
 * - Left rail: active run metadata + stop button
 * - Main: LossChart (large) on top
 * - Middle row: Throughput + VramBar
 * - Bottom row: StreamingViz + LoraHeatmap
 * - VectorWires overlay connecting related panels
 */
import React, { useEffect } from "react";
import { ipc } from "../lib/ipc";
import { useStore } from "../lib/store";
import { LossChart } from "../panels/LossChart";
import { Throughput } from "../panels/Throughput";
import { VramBar } from "../panels/VramBar";
import { StreamingViz } from "../panels/StreamingViz";
import { LoraHeatmap } from "../panels/LoraHeatmap";
import { Panel } from "../chrome/VectorFrame";
import { VectorWires } from "../chrome/VectorWires";
import { fmtLoss, fmtDuration, fmtRelTime } from "../lib/format";

// Wire definitions: which panels should light up together
const MONITOR_WIRES = [
  { from: "panel-loss", to: "panel-throughput", event: "metrics" },
  { from: "panel-throughput", to: "panel-vram", event: "gpu" },
  { from: "panel-vram", to: "panel-streaming", event: "stream_stats" },
  { from: "panel-streaming", to: "panel-lora", event: "metrics" },
];

export function Monitor() {
  const activeRunId = useStore((s) => s.activeRunId);
  const runs = useStore((s) => s.runs);
  const metricsHistory = useStore((s) => s.metricsHistory);
  const appendMetrics = useStore((s) => s.appendMetrics);
  const appendGpu = useStore((s) => s.appendGpu);
  const setStreamStats = useStore((s) => s.setStreamStats);
  const setMemoryStats = useStore((s) => s.setMemoryStats);
  const appendLog = useStore((s) => s.appendLog);
  const upsertRun = useStore((s) => s.upsertRun);
  const setActivePane = useStore((s) => s.setActivePane);

  // Subscribe to IPC events
  useEffect(() => {
    const unsubs = [
      ipc.onMetrics(appendMetrics),
      ipc.onGpu(appendGpu),
      ipc.onStreamStats(setStreamStats),
      ipc.onMemoryStats(setMemoryStats),
      ipc.onLogLine(appendLog),
      ipc.onRunComplete((e) => {
        // Update-only: never insert partial RunSummary objects.
        // If the run doesn't exist locally, skip — run.list will surface it.
        const existing = useStore.getState().runs.find((r) => r.run_id === e.run_id);
        if (existing) {
          upsertRun({
            ...existing,
            status: e.exit_code === 0 ? "complete" : "failed",
            end_time: Date.now(),
          });
        }
      }),
    ];
    return () => unsubs.forEach((u) => u());
  }, [appendMetrics, appendGpu, setStreamStats, setMemoryStats, appendLog, upsertRun]);

  const activeRun = activeRunId ? runs.find((r) => r.run_id === activeRunId) : null;
  const latestMetrics = activeRunId
    ? (metricsHistory[activeRunId] || []).slice(-1)[0]
    : null;

  const handleStop = async () => {
    if (!activeRunId) return;
    await ipc.stopRun(activeRunId);
    const existing = useStore.getState().runs.find((r) => r.run_id === activeRunId);
    if (existing) {
      upsertRun({ ...existing, status: "stopped", end_time: Date.now() });
    }
  };

  return (
    <div
      style={{
        display: "flex",
        height: "100%",
        position: "relative",
        overflow: "hidden",
      }}
    >
      {/* VectorWires overlay */}
      <VectorWires wires={MONITOR_WIRES} />

      {/* Left rail — run metadata */}
      <div
        style={{
          width: 200,
          flexShrink: 0,
          borderRight: "1px solid var(--border)",
          padding: "14px 12px",
          display: "flex",
          flexDirection: "column",
          gap: 12,
          overflowY: "auto",
        }}
      >
        <div
          style={{
            fontSize: "9px",
            fontFamily: "'Share Tech Mono', monospace",
            letterSpacing: "2px",
            color: "var(--dim)",
            textTransform: "uppercase",
          }}
        >
          Active Run
        </div>

        {activeRun ? (
          <>
            {/* Status */}
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <div
                className={`status-dot ${activeRun.status === "running" ? "running" : activeRun.status === "failed" ? "error" : "idle"}`}
              />
              <span
                style={{
                  fontSize: "10px",
                  fontFamily: "'Share Tech Mono', monospace",
                  letterSpacing: "1px",
                  textTransform: "uppercase",
                  color:
                    activeRun.status === "running"
                      ? "var(--lime)"
                      : activeRun.status === "failed"
                      ? "var(--magenta)"
                      : "var(--dim)",
                }}
              >
                {activeRun.status}
              </span>
            </div>

            {/* Model */}
            <div>
              <div style={{ fontSize: "9px", color: "var(--dim)", letterSpacing: "1px", fontFamily: "'Share Tech Mono', monospace" }}>
                MODEL
              </div>
              <div style={{ fontSize: "10px", color: "var(--text)", marginTop: 2, wordBreak: "break-all" }}>
                {activeRun.model.split("/").pop()}
              </div>
            </div>

            {/* Method */}
            <div>
              <div style={{ fontSize: "9px", color: "var(--dim)", letterSpacing: "1px", fontFamily: "'Share Tech Mono', monospace" }}>
                METHOD
              </div>
              <div
                style={{
                  fontSize: "11px",
                  fontFamily: "'Share Tech Mono', monospace",
                  letterSpacing: "1px",
                  textTransform: "uppercase",
                  color: "var(--cyan)",
                  marginTop: 2,
                }}
              >
                {activeRun.method}
              </div>
            </div>

            {/* Progress */}
            {activeRun.steps_total && (
              <div>
                <div style={{ fontSize: "9px", color: "var(--dim)", letterSpacing: "1px", fontFamily: "'Share Tech Mono', monospace" }}>
                  PROGRESS
                </div>
                <div
                  className="num"
                  style={{
                    fontSize: "16px",
                    fontFamily: "'JetBrains Mono', monospace",
                    color: "var(--text)",
                    marginTop: 2,
                  }}
                >
                  {activeRun.steps_done || 0}
                  <span style={{ color: "var(--dim)", fontSize: "12px" }}>
                    /{activeRun.steps_total}
                  </span>
                </div>
                <div
                  style={{
                    height: 3,
                    background: "var(--grid)",
                    marginTop: 4,
                    overflow: "hidden",
                  }}
                >
                  <div
                    style={{
                      height: "100%",
                      width: `${((activeRun.steps_done || 0) / activeRun.steps_total) * 100}%`,
                      background: "var(--cyan)",
                      transition: "width 0.5s ease",
                    }}
                  />
                </div>
              </div>
            )}

            {/* Loss */}
            {latestMetrics && (
              <div>
                <div style={{ fontSize: "9px", color: "var(--dim)", letterSpacing: "1px", fontFamily: "'Share Tech Mono', monospace" }}>
                  LOSS
                </div>
                <div
                  className="num"
                  style={{
                    fontSize: "22px",
                    fontFamily: "'JetBrains Mono', monospace",
                    color: "var(--cyan)",
                    marginTop: 2,
                  }}
                >
                  {fmtLoss(latestMetrics.loss)}
                </div>
              </div>
            )}

            {/* Duration */}
            <div>
              <div style={{ fontSize: "9px", color: "var(--dim)", letterSpacing: "1px", fontFamily: "'Share Tech Mono', monospace" }}>
                ELAPSED
              </div>
              <div
                className="num"
                style={{ fontSize: "12px", fontFamily: "'JetBrains Mono', monospace", color: "var(--text)", marginTop: 2 }}
              >
                {fmtDuration(activeRun.start_time)}
              </div>
            </div>

            {/* Stop button */}
            {activeRun.status === "running" && (
              <button
                onClick={handleStop}
                className="btn-danger"
                style={{ marginTop: "auto" }}
              >
                STOP RUN
              </button>
            )}
          </>
        ) : (
          <div style={{ color: "var(--dim)", fontSize: "10px" }}>
            No active run. Start one from Launch.
          </div>
        )}

        {/* Recent runs list */}
        <div
          style={{
            marginTop: 16,
            paddingTop: 12,
            borderTop: "1px solid var(--border)",
          }}
        >
          <div style={{ fontSize: "9px", color: "var(--dim)", letterSpacing: "2px", fontFamily: "'Share Tech Mono', monospace", marginBottom: 6 }}>
            RECENT
          </div>
          {runs.slice(0, 5).map((r) => (
            <div
              key={r.run_id}
              style={{
                padding: "4px 0",
                cursor: "pointer",
                borderBottom: "1px solid var(--grid)",
                display: "flex",
                alignItems: "center",
                gap: 6,
              }}
              onClick={() => useStore.getState().setActiveRunId(r.run_id)}
            >
              <div
                className={`status-dot ${r.status === "running" ? "running" : r.status === "failed" ? "error" : "idle"}`}
                style={{ width: 6, height: 6 }}
              />
              <span style={{ fontSize: "10px", color: r.run_id === activeRunId ? "var(--cyan)" : "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {r.model.split("/").pop()}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Main panel area */}
      <div
        style={{
          flex: 1,
          display: "grid",
          gridTemplateRows: "2fr 1fr 1fr",
          gridTemplateColumns: "1fr",
          gap: 1,
          padding: 8,
          overflow: "hidden",
          background: "var(--grid)",
        }}
      >
        {/* Row 1: Loss chart */}
        <Panel id="panel-loss" label="Training Loss" style={{ overflow: "hidden", padding: 8 }}>
          <LossChart runId={activeRunId || ""} />
        </Panel>

        {/* Row 2: Throughput + VRAM */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 1 }}>
          <Panel id="panel-throughput" label="Throughput" style={{ overflow: "hidden" }}>
            <Throughput />
          </Panel>
          <Panel id="panel-vram" label="VRAM" style={{ overflow: "hidden" }}>
            <VramBar />
          </Panel>
        </div>

        {/* Row 3: Streaming + LoRA */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 1 }}>
          <Panel id="panel-streaming" label="Layer Streaming" style={{ overflow: "hidden" }}>
            <StreamingViz />
          </Panel>
          <Panel id="panel-lora" label="LoRA Norms" style={{ overflow: "hidden" }}>
            <LoraHeatmap />
          </Panel>
        </div>
      </div>
    </div>
  );
}
