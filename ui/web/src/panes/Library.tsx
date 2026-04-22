/**
 * Library pane — checkpoints and LoRA adapters.
 *
 * Two tabs: Checkpoints | LoRA Adapters
 * Row actions: Open (xdg-open), Merge, Delete (with confirm)
 */
import React, { useState, useEffect, useCallback } from "react";
import { ipc } from "../lib/ipc";
import { useStore } from "../lib/store";
import { Panel } from "../chrome/VectorFrame";
import type { Checkpoint } from "../lib/ipc-types";
import { fmtRelTime } from "../lib/format";

type Tab = "checkpoints" | "lora";

export function Library() {
  const runs = useStore((s) => s.runs);
  const [tab, setTab] = useState<Tab>("checkpoints");
  const [outputDir, setOutputDir] = useState("");
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [loading, setLoading] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);

  const handleLoadCheckpoints = useCallback(async () => {
    if (!outputDir.trim()) return;
    setLoading(true);
    try {
      const ckpts = await ipc.listCheckpoints(outputDir.trim());
      setCheckpoints(ckpts);
    } catch (_e) {
      setCheckpoints([]);
    } finally {
      setLoading(false);
    }
  }, [outputDir]);

  // Auto-fill from active run
  const activeRunId = useStore((s) => s.activeRunId);
  useEffect(() => {
    // Runs don't carry output_dir in summary, but we can guess
    // In real usage the C shell would provide this
  }, [activeRunId, runs]);

  const loraAdapters = checkpoints.filter((c) => c.is_lora);
  const modelCheckpoints = checkpoints.filter((c) => !c.is_lora);

  const displayed = tab === "checkpoints" ? modelCheckpoints : loraAdapters;

  return (
    <div style={{ height: "100%", overflowY: "auto", padding: 16 }}>
      {/* Path input */}
      <Panel style={{ padding: 12, marginBottom: 12 }}>
        <div
          style={{
            fontSize: "9px",
            fontFamily: "'Share Tech Mono', monospace",
            letterSpacing: "2px",
            color: "var(--dim)",
            textTransform: "uppercase",
            marginBottom: 6,
          }}
        >
          Output Directory
        </div>
        <div style={{ display: "flex", gap: 6 }}>
          <input
            style={{
              flex: 1,
              background: "var(--panel-hi)",
              border: "1px solid var(--border)",
              color: "var(--text)",
              padding: "6px 10px",
              fontSize: "12px",
              fontFamily: "inherit",
              outline: "none",
            }}
            placeholder="/path/to/outputs"
            value={outputDir}
            onChange={(e) => setOutputDir(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleLoadCheckpoints()}
          />
          <button onClick={handleLoadCheckpoints} disabled={loading}>
            {loading ? "..." : "SCAN"}
          </button>
        </div>
      </Panel>

      {/* Tabs */}
      <div className="seg-ctrl" style={{ marginBottom: 12 }}>
        {(["checkpoints", "lora"] as Tab[]).map((t) => (
          <button
            key={t}
            className={tab === t ? "active" : ""}
            onClick={() => setTab(t)}
          >
            {t === "checkpoints" ? `CHECKPOINTS (${modelCheckpoints.length})` : `LORA (${loraAdapters.length})`}
          </button>
        ))}
      </div>

      {/* List */}
      <Panel style={{ overflow: "hidden" }}>
        {displayed.length === 0 ? (
          <div
            style={{
              padding: 40,
              textAlign: "center",
              color: "var(--dim)",
              fontFamily: "'Share Tech Mono', monospace",
              fontSize: "11px",
              letterSpacing: "2px",
            }}
          >
            {loading ? "SCANNING..." : "NO ITEMS FOUND"}
          </div>
        ) : (
          displayed.map((ckpt) => (
            <div
              key={ckpt.path}
              style={{
                padding: "10px 14px",
                borderBottom: "1px solid var(--grid)",
                display: "flex",
                alignItems: "center",
                gap: 12,
              }}
            >
              {/* Path */}
              <div style={{ flex: 1, minWidth: 0 }}>
                <div
                  style={{
                    fontSize: "11px",
                    color: "var(--text)",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  }}
                >
                  {ckpt.path.split("/").pop()}
                </div>
                <div style={{ fontSize: "10px", color: "var(--dim)", marginTop: 2 }}>
                  {ckpt.path}
                </div>
              </div>

              {/* Step */}
              {ckpt.step > 0 && (
                <div
                  className="num"
                  style={{
                    fontSize: "11px",
                    fontFamily: "'JetBrains Mono', monospace",
                    color: "var(--amber)",
                    flexShrink: 0,
                  }}
                >
                  step {ckpt.step}
                </div>
              )}

              {/* Loss */}
              {ckpt.loss !== undefined && ckpt.loss > 0 && (
                <div
                  className="num"
                  style={{
                    fontSize: "11px",
                    fontFamily: "'JetBrains Mono', monospace",
                    color: "var(--cyan)",
                    flexShrink: 0,
                  }}
                >
                  {ckpt.loss.toFixed(4)}
                </div>
              )}

              {/* Save time — save_time_s is seconds since epoch; convert to ms for Date */}
              {ckpt.save_time_s && (
                <div style={{ fontSize: "10px", color: "var(--dim)", flexShrink: 0 }}>
                  {fmtRelTime(ckpt.save_time_s * 1000)}
                </div>
              )}

              {/* Actions */}
              <div style={{ display: "flex", gap: 6, flexShrink: 0 }}>
                {ckpt.is_lora && (
                  <button
                    style={{ fontSize: "9px", padding: "3px 8px" }}
                    className="btn-amber"
                    onClick={() => {
                      // Merge action — calls CLI through IPC
                      // For now, show path to user
                      alert(`Merge command:\nmud-puppy --merge-lora ${ckpt.path}`);
                    }}
                  >
                    MERGE
                  </button>
                )}
                <button
                  style={{ fontSize: "9px", padding: "3px 8px" }}
                  onClick={async () => {
                    // xdg-open via a dedicated IPC call (not yet in manifest, stub)
                    console.log("open:", ckpt.path);
                  }}
                >
                  OPEN
                </button>
                <button
                  style={{ fontSize: "9px", padding: "3px 8px" }}
                  className="btn-danger"
                  onClick={() => setConfirmDelete(ckpt.path)}
                >
                  DEL
                </button>
              </div>
            </div>
          ))
        )}
      </Panel>

      {/* Delete confirmation dialog */}
      {confirmDelete && (
        <div
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(0,0,0,0.7)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 100,
          }}
        >
          <Panel style={{ padding: 24, maxWidth: 420 }}>
            <div
              style={{
                fontFamily: "'Share Tech Mono', monospace",
                fontSize: "13px",
                letterSpacing: "2px",
                textTransform: "uppercase",
                color: "var(--magenta)",
                marginBottom: 12,
              }}
            >
              Confirm Delete
            </div>
            <div style={{ fontSize: "11px", color: "var(--text)", marginBottom: 16, wordBreak: "break-all" }}>
              {confirmDelete}
            </div>
            <div style={{ display: "flex", gap: 10 }}>
              <button
                className="btn-danger"
                onClick={() => {
                  // Deletion is USER-only operation; report to user to execute
                  alert(`To delete, run:\nrm -rf "${confirmDelete}"`);
                  setConfirmDelete(null);
                }}
              >
                CONFIRM DELETE
              </button>
              <button onClick={() => setConfirmDelete(null)}>CANCEL</button>
            </div>
          </Panel>
        </div>
      )}
    </div>
  );
}
