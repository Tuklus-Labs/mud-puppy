/**
 * Monitor pane — 240px run summary rail + panels column (1.4fr hero loss
 * chart + 1fr row of Throughput / VRAM / Streaming).
 *
 * Mirrors the 2026-04-22 design handoff. Real metrics from store; when a
 * run is active the panels pull from IPC event history, otherwise they
 * render recent telemetry at zero so the layout stays visible.
 */
import React, { useMemo } from "react";
import { Panel } from "../chrome/VectorFrame";
import { useStore } from "../lib/store";
import { ipc } from "../lib/ipc";
import type { MetricsEvent } from "../lib/ipc-types";
import { fmtDuration, fmtLoss } from "../lib/format";

export function Monitor() {
  const runs = useStore((s) => s.runs);
  const activeRunId = useStore((s) => s.activeRunId);
  const metricsHistory = useStore((s) => s.metricsHistory);
  const gpuHistory = useStore((s) => s.gpuHistory);
  const streamStats = useStore((s) => s.streamStats);
  const memoryStats = useStore((s) => s.memoryStats);

  const run = activeRunId
    ? runs.find((r) => r.run_id === activeRunId) ?? null
    : runs.find((r) => r.status === "running") ?? null;

  const metrics = run ? metricsHistory[run.run_id] ?? [] : [];
  const latest = metrics.length > 0 ? metrics[metrics.length - 1] : null;
  const prevN = metrics.slice(-100)[0];
  const deltaLoss =
    latest && prevN ? latest.loss - prevN.loss : 0;

  const progress =
    run && run.steps_total && run.steps_done != null
      ? (run.steps_done / run.steps_total) * 100
      : 0;

  const tokensPerSec = latest?.tokens_per_sec ?? 0;
  const tputSeries = useMemo(
    () =>
      metrics
        .slice(-60)
        .map((m) => m.tokens_per_sec ?? 0)
        .filter((v) => v > 0),
    [metrics],
  );
  const tputAvg =
    tputSeries.length > 0
      ? tputSeries.reduce((a, b) => a + b, 0) / tputSeries.length
      : 0;
  const tputMax = tputSeries.length > 0 ? Math.max(...tputSeries) : 0;

  const gpuLatest =
    gpuHistory.length > 0 ? gpuHistory[gpuHistory.length - 1] : null;
  const vramUsed = memoryStats?.allocated_gb ?? gpuLatest?.vram_used_gb ?? 0;
  const vramTotal = gpuLatest?.vram_total_gb ?? 24;
  const vramPct = vramTotal > 0 ? (vramUsed / vramTotal) * 100 : 0;

  const stopRun = async () => {
    if (!run) return;
    try {
      await ipc.stopRun(run.run_id);
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className="pane">
      <div className="pane-title">
        <h1>Monitor</h1>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          {run?.status === "running" && (
            <>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <span className="dot running" />
                <span
                  className="mono"
                  style={{ fontSize: "var(--f-xs)", color: "var(--lime)" }}
                >
                  Training
                </span>
              </div>
              {run.steps_total && run.steps_done != null && (
                <div
                  className="num"
                  style={{ fontSize: "var(--f-sm)", color: "var(--text-dim)" }}
                >
                  step {run.steps_done.toLocaleString()} /{" "}
                  {run.steps_total.toLocaleString()}
                </div>
              )}
              <button className="btn btn-danger" onClick={stopRun}>
                Stop run
              </button>
            </>
          )}
        </div>
      </div>

      {!run ? (
        <Panel>
          <div
            style={{
              padding: 60,
              textAlign: "center",
              color: "var(--text-dim)",
            }}
          >
            <div
              className="mono"
              style={{ fontSize: 14, marginBottom: 8, color: "var(--cyan)" }}
            >
              No active run
            </div>
            <div style={{ fontSize: 12 }}>
              Launch a configuration or pick a recent run to populate this
              view.
            </div>
          </div>
        </Panel>
      ) : (
        <div className="monitor-grid">
          <Panel label="Run" bodyStyle={{ padding: "20px 18px" }}>
            <div className="run-summary">
              <div className="run-stat">
                <div className="label">Model</div>
                <div
                  className="v"
                  style={{ fontSize: 13, wordBreak: "break-all" }}
                >
                  {run.model.split("/").pop()}
                </div>
                <div
                  style={{
                    fontSize: 10,
                    color: "var(--dim)",
                    fontFamily: "var(--font-num)",
                  }}
                >
                  {run.model.split("/")[0]}
                </div>
              </div>
              <div className="run-stat">
                <div className="label">Method</div>
                <div
                  className="v cyan"
                  style={{
                    textTransform: "uppercase",
                    fontFamily: "var(--font-heading)",
                    letterSpacing: "0.2em",
                  }}
                >
                  {run.method}
                </div>
              </div>
              <div className="run-stat">
                <div className="label">Current loss</div>
                <div className="v big cyan num">
                  {latest ? fmtLoss(latest.loss) : "—"}
                </div>
                {metrics.length >= 2 && (
                  <Sparkline
                    values={metrics.slice(-40).map((m) => m.loss)}
                    color="var(--cyan)"
                    height={30}
                  />
                )}
              </div>
              <div className="run-stat">
                <div className="label">Progress</div>
                <div className="v num">
                  {run.steps_done ?? 0}
                  <span style={{ color: "var(--dim)", fontSize: 12 }}>
                    {" "}
                    / {run.steps_total ?? "?"}
                  </span>
                </div>
                <div className="progress">
                  <div style={{ width: `${progress}%` }} />
                </div>
                <div
                  style={{
                    fontSize: 10,
                    color: "var(--dim)",
                    fontFamily: "var(--font-num)",
                  }}
                >
                  {progress.toFixed(1)}% complete
                </div>
              </div>
              <div className="run-stat">
                <div className="label">Elapsed</div>
                <div className="v num">
                  {fmtDuration(run.start_time, run.end_time)}
                </div>
              </div>
              <div className="run-stat">
                <div className="label">Run ID</div>
                <div
                  className="v num"
                  style={{ fontSize: 12, color: "var(--text-dim)" }}
                >
                  {run.run_id}
                </div>
              </div>
            </div>
          </Panel>

          <div className="monitor-panels">
            <Panel
              label="Training Loss"
              bodyStyle={{
                display: "flex",
                flexDirection: "column",
                height: "100%",
                padding: "24px 24px 20px",
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  marginBottom: 18,
                  flexWrap: "wrap",
                  gap: 14,
                }}
              >
                <div
                  style={{
                    display: "flex",
                    gap: 28,
                    alignItems: "baseline",
                    flexWrap: "wrap",
                  }}
                >
                  <div>
                    <div className="label">Current</div>
                    <div
                      className="num"
                      style={{
                        fontSize: 28,
                        color: "var(--cyan)",
                        lineHeight: 1.1,
                      }}
                    >
                      {latest ? fmtLoss(latest.loss) : "—"}
                    </div>
                  </div>
                  <div>
                    <div className="label">Δ 100</div>
                    <div
                      className="num"
                      style={{
                        fontSize: 18,
                        color:
                          deltaLoss < 0 ? "var(--lime)" : "var(--magenta)",
                        lineHeight: 1.1,
                      }}
                    >
                      {deltaLoss >= 0 ? "+" : "−"}
                      {Math.abs(deltaLoss).toFixed(3)}
                    </div>
                  </div>
                  <div>
                    <div className="label">Smoothed</div>
                    <div
                      className="num"
                      style={{ fontSize: 18, color: "var(--text)" }}
                    >
                      {latest
                        ? fmtLoss(
                            metrics
                              .slice(-20)
                              .reduce((s, m) => s + m.loss, 0) /
                              Math.min(metrics.length, 20),
                          )
                        : "—"}
                    </div>
                  </div>
                </div>
                <div className="seg">
                  <button className="active">Loss</button>
                  <button>LR</button>
                  <button>Grad</button>
                </div>
              </div>
              <div style={{ flex: 1, minHeight: 220 }}>
                <LossChart data={metrics} />
              </div>
            </Panel>

            <div className="monitor-sub">
              <Panel
                label="Throughput"
                bodyStyle={{
                  padding: 20,
                  height: "100%",
                  display: "flex",
                  flexDirection: "column",
                }}
              >
                <div className="label">Tokens / sec</div>
                <div
                  className="num"
                  style={{
                    fontSize: 32,
                    color: "var(--lime)",
                    lineHeight: 1.1,
                    marginTop: 4,
                  }}
                >
                  {Math.round(tokensPerSec).toLocaleString()}
                </div>
                <div style={{ flex: 1, minHeight: 70, marginTop: 10 }}>
                  <Sparkline
                    values={tputSeries.length ? tputSeries : [0]}
                    color="var(--lime)"
                    height="100%"
                  />
                </div>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    fontSize: 11,
                    color: "var(--dim)",
                    fontFamily: "var(--font-num)",
                    marginTop: 8,
                  }}
                >
                  <span>avg {Math.round(tputAvg).toLocaleString()}</span>
                  <span>max {Math.round(tputMax).toLocaleString()}</span>
                </div>
              </Panel>

              <Panel
                label="VRAM"
                bodyStyle={{
                  padding: 20,
                  height: "100%",
                  display: "flex",
                  flexDirection: "column",
                }}
              >
                <div className="label">Allocated</div>
                <div
                  className="num"
                  style={{
                    fontSize: 32,
                    color:
                      vramPct > 85 ? "var(--amber)" : "var(--cyan)",
                    lineHeight: 1.1,
                    marginTop: 4,
                  }}
                >
                  {vramUsed.toFixed(1)}
                  <span style={{ fontSize: 16, color: "var(--dim)" }}>
                    {" "}
                    / {vramTotal.toFixed(1)} GB
                  </span>
                </div>
                <div className="vram-bar" style={{ marginTop: 14, height: 8 }}>
                  <div
                    style={{
                      width: `${vramPct}%`,
                      background:
                        vramPct > 85 ? "var(--amber)" : "var(--cyan)",
                    }}
                  />
                </div>
                <div style={{ flex: 1 }} />
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    fontSize: 11,
                    color: "var(--dim)",
                    fontFamily: "var(--font-num)",
                    marginTop: 10,
                  }}
                >
                  <span>
                    Reserved{" "}
                    {(memoryStats?.reserved_gb ?? 0).toFixed(1)} GB
                  </span>
                  <span>
                    Frag{" "}
                    {((memoryStats?.fragmentation ?? 0) * 100).toFixed(1)}%
                  </span>
                </div>
              </Panel>

              <Panel
                label="Layer Streaming"
                bodyStyle={{
                  padding: 20,
                  height: "100%",
                  display: "flex",
                  flexDirection: "column",
                }}
              >
                <div className="label">
                  Layer cache ·{" "}
                  {streamStats?.layers_total ?? 0} layers
                </div>
                <StreamingRows stats={streamStats} />
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    fontSize: 11,
                    color: "var(--dim)",
                    fontFamily: "var(--font-num)",
                    marginTop: 12,
                  }}
                >
                  <span
                    style={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: 4,
                    }}
                  >
                    <span
                      className="dot"
                      style={{
                        background: "var(--cyan)",
                        width: 6,
                        height: 6,
                      }}
                    />
                    active
                  </span>
                  <span
                    style={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: 4,
                    }}
                  >
                    <span
                      className="dot"
                      style={{
                        background: "var(--amber)",
                        width: 6,
                        height: 6,
                      }}
                    />
                    evicting
                  </span>
                  <span className="num">
                    {((streamStats?.prefetch_hit_rate ?? 0) * 100).toFixed(0)}% hit
                  </span>
                </div>
              </Panel>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────── Inline panels ───────────────────────────

function LossChart({ data }: { data: MetricsEvent[] }) {
  const W = 900;
  const H = 280;
  const padL = 46;
  const padR = 20;
  const padT = 16;
  const padB = 28;

  if (data.length < 2) {
    return (
      <svg
        className="chart"
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        style={{ width: "100%", height: "100%" }}
      >
        <text
          x={W / 2}
          y={H / 2}
          textAnchor="middle"
          fontFamily="var(--font-heading)"
          fontSize="13"
          letterSpacing="3"
          fill="var(--dim)"
        >
          AWAITING METRICS
        </text>
      </svg>
    );
  }

  const xs = data.map((d) => d.step);
  const ys = data.map((d) => d.loss);
  const xmin = Math.min(...xs);
  const xmax = Math.max(...xs);
  const ymin = Math.min(...ys) * 0.9;
  const ymax = Math.max(...ys) * 1.05;
  const x = (v: number) =>
    padL + ((v - xmin) / Math.max(1, xmax - xmin)) * (W - padL - padR);
  const y = (v: number) =>
    padT + (1 - (v - ymin) / Math.max(1e-9, ymax - ymin)) * (H - padT - padB);

  const path = data
    .map(
      (d, i) =>
        `${i ? "L" : "M"}${x(d.step).toFixed(1)},${y(d.loss).toFixed(1)}`,
    )
    .join(" ");
  const latest = data[data.length - 1];

  return (
    <svg
      className="chart"
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="none"
      style={{ width: "100%", height: "100%", display: "block" }}
    >
      <defs>
        <linearGradient id="lossFill" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor="rgba(0,229,255,0.35)" />
          <stop offset="100%" stopColor="rgba(0,229,255,0)" />
        </linearGradient>
      </defs>
      {[0.25, 0.5, 0.75].map((t) => (
        <line
          key={t}
          x1={padL}
          x2={W - padR}
          y1={padT + t * (H - padT - padB)}
          y2={padT + t * (H - padT - padB)}
          stroke="var(--grid)"
          strokeWidth="1"
          strokeDasharray="2 4"
        />
      ))}
      {[ymax, (ymax + ymin) / 2, ymin].map((v, i) => (
        <text
          key={i}
          x={padL - 6}
          y={y(v) + 3}
          textAnchor="end"
          fontFamily="var(--font-num)"
          fontSize="12"
          fill="var(--dim)"
        >
          {v.toFixed(2)}
        </text>
      ))}
      {[0, 0.5, 1].map((t, i) => {
        const step = xmin + t * (xmax - xmin);
        return (
          <text
            key={i}
            x={padL + t * (W - padL - padR)}
            y={H - 6}
            textAnchor={t === 0 ? "start" : t === 1 ? "end" : "middle"}
            fontFamily="var(--font-num)"
            fontSize="12"
            fill="var(--dim)"
          >
            step {Math.round(step)}
          </text>
        );
      })}
      <path
        d={`${path} L${x(xmax)},${H - padB} L${x(xmin)},${H - padB} Z`}
        fill="url(#lossFill)"
      />
      <path
        d={path}
        fill="none"
        stroke="var(--cyan)"
        strokeWidth="1.8"
        style={{ filter: "drop-shadow(0 0 3px var(--cyan))" }}
      />
      <circle cx={x(latest.step)} cy={y(latest.loss)} r="3" fill="var(--cyan)" />
      <circle
        cx={x(latest.step)}
        cy={y(latest.loss)}
        r="6"
        fill="none"
        stroke="var(--cyan)"
        strokeWidth="0.8"
        opacity="0.5"
      />
    </svg>
  );
}

function Sparkline({
  values,
  color = "var(--lime)",
  height = 60,
}: {
  values: number[];
  color?: string;
  height?: number | "100%";
}) {
  const W = 240;
  const H = typeof height === "number" ? height : 60;
  if (values.length < 2) {
    return (
      <svg
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        style={{
          width: "100%",
          height: typeof height === "number" ? height : "100%",
          display: "block",
        }}
      />
    );
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const path = values
    .map((v, i) => {
      const xx = (i / (values.length - 1)) * W;
      const yy = H - ((v - min) / range) * (H - 4) - 2;
      return `${i ? "L" : "M"}${xx.toFixed(1)},${yy.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="none"
      style={{
        width: "100%",
        height: typeof height === "number" ? height : "100%",
        display: "block",
      }}
    >
      <path d={path} fill="none" stroke={color} strokeWidth="1.5" />
      <path
        d={`${path} L${W},${H} L0,${H} Z`}
        fill={color}
        opacity="0.12"
      />
    </svg>
  );
}

function StreamingRows({
  stats,
}: {
  stats: ReturnType<typeof useStore.getState>["streamStats"];
}) {
  // Up to 3 rows of 11 cells each — matches the design.
  const total = stats?.layers_total ?? 33;
  const resident = stats?.layers_resident ?? 0;
  const layerStates = stats?.layer_states ?? [];

  const cellState = (idx: number): string => {
    const ls = layerStates.find((s) => s.idx === idx);
    if (ls?.prefetching) return "evicting";
    if (ls?.resident) return "hot";
    if (idx < resident) return "warm";
    return "";
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 6,
        marginTop: 12,
        flex: 1,
        justifyContent: "center",
      }}
    >
      {[0, 1, 2].map((row) => (
        <div className="stream-row" key={row}>
          <span className="stream-label" style={{ fontSize: 11, width: 32 }}>
            {String(row * 11).padStart(2, "0")}
          </span>
          <div className="stream-cells">
            {Array.from({ length: 11 }).map((_, i) => {
              const idx = row * 11 + i;
              if (idx >= total)
                return <span key={i} style={{ flex: 1, height: 14 }} />;
              return (
                <span
                  key={i}
                  className={`stream-cell ${cellState(idx)}`}
                  style={{ height: 14 }}
                />
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
