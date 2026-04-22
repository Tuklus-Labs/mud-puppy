/**
 * Runs pane — history and comparison of past runs.
 *
 * Features:
 * - List of runs with name, method, model, final loss, duration, date
 * - Click to navigate to Monitor for that run
 * - Checkbox select for two-run compare overlay on LossChart
 */
import React, { useEffect, useState } from "react";
import { ipc } from "../lib/ipc";
import { useStore } from "../lib/store";
import { Panel } from "../chrome/VectorFrame";
import { fmtLoss, fmtDuration, fmtRelTime } from "../lib/format";

export function Runs() {
  const runs = useStore((s) => s.runs);
  const setRuns = useStore((s) => s.setRuns);
  const activeRunId = useStore((s) => s.activeRunId);
  const setActiveRunId = useStore((s) => s.setActiveRunId);
  const compareRunId = useStore((s) => s.compareRunId);
  const setCompareRunId = useStore((s) => s.setCompareRunId);
  const setActivePane = useStore((s) => s.setActivePane);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    ipc.listRuns().then((rs) => {
      setRuns(rs);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, [setRuns]);

  const handleSelect = (runId: string) => {
    setActiveRunId(runId);
    setActivePane("monitor");
  };

  const handleCompareToggle = (runId: string) => {
    if (compareRunId === runId) {
      setCompareRunId(null);
    } else {
      setCompareRunId(runId);
    }
  };

  if (loading) {
    return (
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          height: "100%",
          color: "var(--dim)",
          fontFamily: "'Share Tech Mono', monospace",
          fontSize: "11px",
          letterSpacing: "2px",
        }}
      >
        LOADING RUNS...
      </div>
    );
  }

  return (
    <div style={{ height: "100%", overflowY: "auto", padding: 16 }}>
      <Panel
        label={`Runs (${runs.length})`}
        style={{ height: "100%", overflow: "hidden" }}
      >
        {/* Header row */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "24px 1fr 80px 80px 70px 80px 32px",
            gap: 8,
            padding: "8px 12px",
            borderBottom: "1px solid var(--border)",
            fontSize: "9px",
            fontFamily: "'Share Tech Mono', monospace",
            letterSpacing: "2px",
            textTransform: "uppercase",
            color: "var(--dim)",
          }}
        >
          <span></span>
          <span>Model</span>
          <span>Method</span>
          <span>Loss</span>
          <span>Duration</span>
          <span>Started</span>
          <span>CMP</span>
        </div>

        {runs.length === 0 ? (
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              padding: 40,
              color: "var(--dim)",
              fontFamily: "'Share Tech Mono', monospace",
              fontSize: "11px",
              letterSpacing: "2px",
            }}
          >
            NO RUNS YET
          </div>
        ) : (
          <div style={{ overflowY: "auto", height: "calc(100% - 40px)" }}>
            {runs.map((run) => {
              const isActive = run.run_id === activeRunId;
              const isCompare = run.run_id === compareRunId;

              return (
                <div
                  key={run.run_id}
                  style={{
                    display: "grid",
                    gridTemplateColumns: "24px 1fr 80px 80px 70px 80px 32px",
                    gap: 8,
                    padding: "8px 12px",
                    borderBottom: "1px solid var(--grid)",
                    cursor: "pointer",
                    background: isActive ? "rgba(0,229,255,0.06)" : "transparent",
                    transition: "background 0.15s",
                    alignItems: "center",
                  }}
                  onMouseEnter={(e) => {
                    if (!isActive) e.currentTarget.style.background = "rgba(255,255,255,0.03)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = isActive ? "rgba(0,229,255,0.06)" : "transparent";
                  }}
                  onClick={() => handleSelect(run.run_id)}
                >
                  {/* Status dot */}
                  <div style={{ display: "flex", justifyContent: "center" }}>
                    <div
                      className={`status-dot ${run.status === "running" ? "running" : run.status === "failed" ? "error" : "idle"}`}
                      style={{ width: 7, height: 7 }}
                    />
                  </div>

                  {/* Model */}
                  <div
                    style={{
                      fontSize: "11px",
                      color: isActive ? "var(--cyan)" : "var(--text)",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {run.model}
                  </div>

                  {/* Method */}
                  <div
                    style={{
                      fontSize: "10px",
                      fontFamily: "'Share Tech Mono', monospace",
                      letterSpacing: "1px",
                      textTransform: "uppercase",
                      color: "var(--amber)",
                    }}
                  >
                    {run.method}
                  </div>

                  {/* Loss */}
                  <div
                    className="num"
                    style={{
                      fontSize: "11px",
                      fontFamily: "'JetBrains Mono', monospace",
                      color: run.final_loss ? "var(--lime)" : "var(--dim)",
                    }}
                  >
                    {run.final_loss ? fmtLoss(run.final_loss) : "--"}
                  </div>

                  {/* Duration */}
                  <div
                    className="num"
                    style={{
                      fontSize: "10px",
                      fontFamily: "'JetBrains Mono', monospace",
                      color: "var(--dim)",
                    }}
                  >
                    {fmtDuration(run.start_time, run.end_time)}
                  </div>

                  {/* Started */}
                  <div
                    style={{ fontSize: "10px", color: "var(--dim)" }}
                  >
                    {fmtRelTime(run.start_time)}
                  </div>

                  {/* Compare checkbox */}
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "center",
                    }}
                    onClick={(e) => {
                      e.stopPropagation();
                      handleCompareToggle(run.run_id);
                    }}
                  >
                    <div
                      style={{
                        width: 14,
                        height: 14,
                        border: `1px solid ${isCompare ? "var(--magenta)" : "var(--border)"}`,
                        background: isCompare ? "rgba(255,43,214,0.2)" : "transparent",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                      }}
                    >
                      {isCompare && (
                        <div style={{ width: 6, height: 6, background: "var(--magenta)" }} />
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </Panel>

      {compareRunId && (
        <div
          style={{
            marginTop: 8,
            padding: "6px 10px",
            background: "rgba(255,43,214,0.1)",
            border: "1px solid var(--magenta)",
            fontSize: "10px",
            fontFamily: "'Share Tech Mono', monospace",
            letterSpacing: "1px",
            color: "var(--magenta)",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <span>COMPARE MODE: {compareRunId.slice(0, 20)}</span>
          <button
            onClick={() => setCompareRunId(null)}
            style={{
              fontSize: "9px",
              padding: "2px 8px",
              borderColor: "var(--magenta)",
              color: "var(--magenta)",
            }}
          >
            CLEAR
          </button>
        </div>
      )}
    </div>
  );
}
