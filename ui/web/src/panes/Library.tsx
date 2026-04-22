/**
 * Library pane — LoRA adapters / checkpoints / merged models listing with
 * path input, scan button, tabbed segmented control, and per-row actions.
 *
 * Grid row layout from the 2026-04-22 design handoff:
 *   20px icon · 1fr name/path · step · loss · rel-time · actions
 */
import React, { useState } from "react";
import { Panel } from "../chrome/VectorFrame";
import { ipc } from "../lib/ipc";
import type { Checkpoint } from "../lib/ipc-types";
import { fmtRelTime, fmtLoss } from "../lib/format";

type Tab = "lora" | "ckpt" | "merged";

export function Library() {
  const [tab, setTab] = useState<Tab>("lora");
  const [outputDir, setOutputDir] = useState("outputs/llama3-finetune");
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [loading, setLoading] = useState(false);

  const scan = async () => {
    if (!outputDir.trim()) return;
    setLoading(true);
    try {
      const r = await ipc.listCheckpoints(outputDir.trim());
      setCheckpoints(Array.isArray(r) ? r : []);
    } catch {
      setCheckpoints([]);
    } finally {
      setLoading(false);
    }
  };

  const loraItems = checkpoints.filter((c) => c.is_lora);
  const ckptItems = checkpoints.filter((c) => !c.is_lora);
  const displayed =
    tab === "lora" ? loraItems : tab === "ckpt" ? ckptItems : [];

  return (
    <div className="pane">
      <div className="pane-title">
        <h1>Library</h1>
        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <input
            style={{ width: 300 }}
            placeholder="/path/to/outputs"
            value={outputDir}
            onChange={(e) => setOutputDir(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && scan()}
          />
          <button className="btn btn-ghost" onClick={scan} disabled={loading}>
            {loading ? "..." : "Scan"}
          </button>
        </div>
      </div>

      <div style={{ display: "flex", gap: 10, marginBottom: 16 }}>
        <div className="seg">
          <button
            className={tab === "lora" ? "active" : ""}
            onClick={() => setTab("lora")}
          >
            LoRA adapters ({loraItems.length})
          </button>
          <button
            className={tab === "ckpt" ? "active" : ""}
            onClick={() => setTab("ckpt")}
          >
            Checkpoints ({ckptItems.length})
          </button>
          <button
            className={tab === "merged" ? "active" : ""}
            onClick={() => setTab("merged")}
          >
            Merged models (0)
          </button>
        </div>
      </div>

      <Panel>
        <div className="lib-list">
          {displayed.length === 0 && (
            <div
              style={{
                padding: 40,
                textAlign: "center",
                color: "var(--text-dim)",
                fontFamily: "var(--font-heading)",
                fontSize: 11,
                letterSpacing: "2px",
              }}
            >
              {loading
                ? "SCANNING..."
                : tab === "merged"
                ? "NO MERGED MODELS YET"
                : "NO ITEMS FOUND. RUN SCAN TO REFRESH."}
            </div>
          )}
          {displayed.map((c) => {
            const base = c.path.split("/").pop() || c.path;
            return (
              <div key={c.path} className="lib-row">
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                  <rect
                    x="1.5"
                    y="2.5"
                    width="13"
                    height="11"
                    stroke="var(--cyan)"
                    strokeWidth="0.8"
                    fill="none"
                  />
                  <line
                    x1="1.5"
                    y1="6"
                    x2="14.5"
                    y2="6"
                    stroke="var(--cyan)"
                    strokeWidth="0.8"
                  />
                  <circle cx="4" cy="4.25" r="0.6" fill="var(--amber)" />
                </svg>
                <div>
                  <div className="lib-name">{base}</div>
                  <div className="lib-path">{c.path}</div>
                </div>
                <div
                  className="num"
                  style={{ color: "var(--amber)", fontSize: 12 }}
                >
                  step {c.step}
                </div>
                <div
                  className="num"
                  style={{ color: "var(--cyan)", fontSize: 12 }}
                >
                  {c.loss != null ? `loss ${fmtLoss(c.loss)}` : "—"}
                </div>
                <div
                  className="num"
                  style={{ color: "var(--dim)", fontSize: 11 }}
                >
                  {c.save_time_s != null
                    ? fmtRelTime(c.save_time_s * 1000)
                    : "—"}
                </div>
                <div style={{ display: "flex", gap: 6 }}>
                  {c.is_lora && (
                    <button
                      className="btn btn-amber"
                      style={{ padding: "4px 10px", fontSize: 9 }}
                      onClick={() =>
                        alert(`Merge command:\nmud-puppy --merge-lora ${c.path}`)
                      }
                    >
                      Merge
                    </button>
                  )}
                  <button
                    className="btn btn-ghost"
                    style={{ padding: "4px 10px", fontSize: 9 }}
                    onClick={() => console.log("open:", c.path)}
                  >
                    Open
                  </button>
                  <button
                    className="btn btn-danger"
                    style={{ padding: "4px 10px", fontSize: 9 }}
                    onClick={() =>
                      alert(`To delete, run:\nrm -rf "${c.path}"`)
                    }
                  >
                    Del
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </Panel>
    </div>
  );
}
