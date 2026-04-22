/**
 * Runs pane — proper <table> with method chips, status dots, compare
 * checkboxes, and active-run highlighting that does NOT cross text (fixed
 * in the 2026-04-22 design handoff by using border cells instead of
 * :first-child inset-shadow).
 */
import React, { useEffect, useMemo, useState } from "react";
import { Panel } from "../chrome/VectorFrame";
import { ipc } from "../lib/ipc";
import { useStore } from "../lib/store";
import { fmtDuration, fmtRelTime, fmtLoss } from "../lib/format";

type StatusFilter = "all" | "running" | "complete" | "failed";

export function Runs() {
  const runs = useStore((s) => s.runs);
  const setRuns = useStore((s) => s.setRuns);
  const setActiveRunId = useStore((s) => s.setActiveRunId);
  const setActivePane = useStore((s) => s.setActivePane);

  const [filter, setFilter] = useState<StatusFilter>("all");
  const [compareSet, setCompareSet] = useState<Set<string>>(new Set());

  useEffect(() => {
    let alive = true;
    ipc
      .listRuns()
      .then((r) => {
        if (alive && Array.isArray(r)) setRuns(r);
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [setRuns]);

  const filtered = useMemo(() => {
    if (filter === "all") return runs;
    return runs.filter((r) => r.status === filter);
  }, [runs, filter]);

  const activeCount = runs.filter((r) => r.status === "running").length;

  const toggleCompare = (id: string) => {
    setCompareSet((s) => {
      const n = new Set(s);
      if (n.has(id)) n.delete(id);
      else n.add(id);
      return n;
    });
  };

  const openRun = (run_id: string) => {
    setActiveRunId(run_id);
    setActivePane("monitor");
  };

  return (
    <div className="pane">
      <div className="pane-title">
        <h1>Runs</h1>
        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <span
            className="mono"
            style={{ fontSize: "var(--f-micro)", color: "var(--dim)" }}
          >
            {runs.length} runs · {activeCount} active
          </span>
          <div className="seg">
            {(["all", "running", "complete", "failed"] as const).map((s) => (
              <button
                key={s}
                className={filter === s ? "active" : ""}
                onClick={() => setFilter(s)}
              >
                {s}
              </button>
            ))}
          </div>
          <button className="btn btn-ghost" disabled={compareSet.size !== 2}>
            Compare ({compareSet.size})
          </button>
        </div>
      </div>

      <Panel>
        <table className="runs-table">
          <thead>
            <tr>
              <th style={{ width: 24 }}></th>
              <th>Run</th>
              <th>Model</th>
              <th>Method</th>
              <th>Final loss</th>
              <th>Steps</th>
              <th>Duration</th>
              <th>Started</th>
              <th style={{ width: 40, textAlign: "center" }}>CMP</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((r) => {
              const active = r.status === "running";
              const stepsCell =
                r.status === "running"
                  ? `${r.steps_done ?? 0}/${r.steps_total ?? "?"}`
                  : r.status === "failed"
                  ? "— (error)"
                  : "complete";
              const durMs =
                r.end_time != null
                  ? r.end_time - r.start_time
                  : Date.now() - r.start_time;
              return (
                <tr
                  key={r.run_id}
                  className={active ? "active" : ""}
                  onClick={() => openRun(r.run_id)}
                >
                  <td>
                    <span
                      className={`dot ${
                        r.status === "running"
                          ? "running"
                          : r.status === "failed"
                          ? "error"
                          : "idle"
                      }`}
                    />
                  </td>
                  <td className="num" style={{ color: "var(--text-dim)" }}>
                    {r.run_id}
                  </td>
                  <td>{r.model}</td>
                  <td>
                    <span className={`method-chip ${r.method}`}>
                      {r.method}
                    </span>
                  </td>
                  <td
                    className="num"
                    style={{
                      color:
                        r.final_loss != null ? "var(--lime)" : "var(--dim)",
                    }}
                  >
                    {r.final_loss != null ? fmtLoss(r.final_loss) : "—"}
                  </td>
                  <td className="num dim">{stepsCell}</td>
                  <td className="num dim">
                    {durMs > 0 ? fmtDuration(0, durMs) : "—"}
                  </td>
                  <td className="dim">{fmtRelTime(r.start_time)}</td>
                  <td
                    style={{ textAlign: "center" }}
                    onClick={(e) => {
                      e.stopPropagation();
                      toggleCompare(r.run_id);
                    }}
                  >
                    <span
                      className={`chk ${compareSet.has(r.run_id) ? "on" : ""}`}
                      style={{
                        borderColor: compareSet.has(r.run_id)
                          ? "var(--magenta)"
                          : undefined,
                        background: compareSet.has(r.run_id)
                          ? "rgba(255,43,214,0.18)"
                          : undefined,
                      }}
                    />
                  </td>
                </tr>
              );
            })}
            {filtered.length === 0 && (
              <tr>
                <td
                  colSpan={9}
                  style={{
                    padding: 40,
                    textAlign: "center",
                    color: "var(--text-dim)",
                  }}
                >
                  No runs match the current filter.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </Panel>
    </div>
  );
}
