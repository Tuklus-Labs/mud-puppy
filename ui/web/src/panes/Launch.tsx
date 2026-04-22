/**
 * Launch pane — configure and start a training run.
 *
 * Sections:
 * - Model picker (HF search + local path)
 * - Dataset picker (path + preview)
 * - Method (segmented control)
 * - Hyperparams
 * - Live VRAM estimate
 * - Start button
 */
import React, { useState, useCallback, useRef } from "react";
import { ipc } from "../lib/ipc";
import { useStore } from "../lib/store";
import type { TrainingConfig } from "../lib/ipc-types";
import type { HFModel } from "../lib/ipc-types";
import { Panel } from "../chrome/VectorFrame";

const METHODS = ["lora", "qlora", "full", "dpo", "grpo", "orpo", "kto"] as const;
type Method = (typeof METHODS)[number];

// Rough VRAM heuristic
function estimateVram(config: Partial<TrainingConfig>): number {
  const modelName = (config.model_name_or_path || "").toLowerCase();
  let params = 7; // Default 7B
  if (modelName.includes("1b") || modelName.includes("tinyllama")) params = 1.1;
  else if (modelName.includes("3b")) params = 3;
  else if (modelName.includes("7b") || modelName.includes("mistral")) params = 7;
  else if (modelName.includes("13b")) params = 13;
  else if (modelName.includes("34b")) params = 34;
  else if (modelName.includes("70b")) params = 70;

  const dtype = config.finetuning_method === "qlora" ? 0.5 : 2; // bf16 = 2 bytes/param
  const modelGb = (params * 1e9 * dtype) / 1e9;

  // Optimizer: AdamW = 2x model for fp32 states (rough)
  const optGb = config.finetuning_method === "full" ? modelGb * 2 : modelGb * 0.1;

  // Activations estimate
  const seqLen = config.max_seq_length || 2048;
  const bsz = config.batch_size || 1;
  const actGb = (seqLen * bsz * 4096 * 4) / 1e9; // rough

  return Math.round((modelGb + optGb + actGb) * 10) / 10;
}

export function Launch() {
  const setActivePane = useStore((s) => s.setActivePane);
  const setActiveRunId = useStore((s) => s.setActiveRunId);
  const upsertRun = useStore((s) => s.upsertRun);

  const [config, setConfig] = useState<Partial<TrainingConfig>>({
    finetuning_method: "lora",
    num_epochs: 1,
    batch_size: 1,
    gradient_accumulation_steps: 8,
    learning_rate: 2e-4,
    max_seq_length: 2048,
    lora_r: 16,
    lora_alpha: 32,
    lora_dropout: 0.05,
    pack_sequences: false,
    stream: false,
    prefetch_layers: 2,
    compile: false,
    zero_offload: false,
    monitor: true,
  });

  const [modelQuery, setModelQuery] = useState("");
  const [modelResults, setModelResults] = useState<HFModel[]>([]);
  const [modelSearching, setModelSearching] = useState(false);
  const [datasetPreview, setDatasetPreview] = useState<{ format: string; rows: unknown[] } | null>(null);
  const [datasetPreviewing, setDatasetPreviewing] = useState(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const searchTimeout = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const handleModelSearch = useCallback((q: string) => {
    setModelQuery(q);
    if (searchTimeout.current) clearTimeout(searchTimeout.current);
    if (q.trim().length < 2) {
      setModelResults([]);
      return;
    }
    searchTimeout.current = setTimeout(async () => {
      setModelSearching(true);
      try {
        const results = await ipc.searchModels(q.trim());
        setModelResults(results);
      } catch (_e) {
        setModelResults([]);
      } finally {
        setModelSearching(false);
      }
    }, 400);
  }, []);

  const handlePreviewDataset = useCallback(async () => {
    if (!config.dataset_path) return;
    setDatasetPreviewing(true);
    try {
      const preview = await ipc.previewDataset(config.dataset_path, 5);
      setDatasetPreview(preview);
    } catch (e) {
      setDatasetPreview(null);
    } finally {
      setDatasetPreviewing(false);
    }
  }, [config.dataset_path]);

  const handleStart = useCallback(async () => {
    if (!config.model_name_or_path || !config.dataset_path || !config.output_dir) {
      setError("Model, dataset, and output dir are required.");
      return;
    }
    setError(null);
    setStarting(true);
    try {
      const handle = await ipc.startRun(config as TrainingConfig);
      upsertRun({
        run_id: handle.run_id,
        model: config.model_name_or_path || "",
        method: config.finetuning_method || "lora",
        dataset: config.dataset_path || "",
        status: "running",
        start_time: Date.now(),
      });
      setActiveRunId(handle.run_id);
      setActivePane("monitor");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to start run");
    } finally {
      setStarting(false);
    }
  }, [config, upsertRun, setActiveRunId, setActivePane]);

  const vramEst = estimateVram(config);
  const vramColor = vramEst > 22 ? "var(--magenta)" : vramEst > 16 ? "var(--amber)" : "var(--lime)";

  const inputStyle: React.CSSProperties = {
    width: "100%",
    background: "var(--panel-hi)",
    border: "1px solid var(--border)",
    color: "var(--text)",
    padding: "6px 10px",
    fontSize: "12px",
    fontFamily: "inherit",
    outline: "none",
  };

  const labelStyle: React.CSSProperties = {
    display: "block",
    fontSize: "9px",
    fontFamily: "'Share Tech Mono', monospace",
    letterSpacing: "2px",
    textTransform: "uppercase",
    color: "var(--dim)",
    marginBottom: 4,
  };

  const fieldStyle: React.CSSProperties = { marginBottom: 14 };

  return (
    <div
      style={{
        height: "100%",
        overflowY: "auto",
        padding: "16px",
        display: "flex",
        flexDirection: "column",
        gap: 16,
      }}
    >
      {/* Model picker */}
      <Panel label="Model" style={{ padding: 14 }}>
        <div style={fieldStyle}>
          <label style={labelStyle}>Model ID or Local Path</label>
          <input
            style={inputStyle}
            value={config.model_name_or_path || ""}
            placeholder="meta-llama/Llama-3-8B"
            onChange={(e) => {
              setConfig((c) => ({ ...c, model_name_or_path: e.target.value }));
              handleModelSearch(e.target.value);
            }}
          />
          {/* Search results dropdown */}
          {modelResults.length > 0 && (
            <div
              style={{
                background: "var(--panel-hi)",
                border: "1px solid var(--border)",
                marginTop: 2,
                maxHeight: 160,
                overflowY: "auto",
                zIndex: 20,
                position: "relative",
              }}
            >
              {modelResults.map((m) => (
                <div
                  key={m.id}
                  style={{
                    padding: "6px 10px",
                    cursor: "pointer",
                    fontSize: "11px",
                    display: "flex",
                    justifyContent: "space-between",
                    borderBottom: "1px solid var(--grid)",
                  }}
                  onMouseEnter={(e) =>
                    (e.currentTarget.style.background = "rgba(0,229,255,0.07)")
                  }
                  onMouseLeave={(e) => (e.currentTarget.style.background = "")}
                  onClick={() => {
                    setConfig((c) => ({ ...c, model_name_or_path: m.id }));
                    setModelResults([]);
                  }}
                >
                  <span style={{ color: "var(--text)" }}>{m.id}</span>
                  <span style={{ color: "var(--dim)", fontSize: "10px" }}>
                    {(m.downloads / 1000).toFixed(0)}k DL
                  </span>
                </div>
              ))}
            </div>
          )}
          {modelSearching && (
            <div style={{ fontSize: "10px", color: "var(--dim)", marginTop: 4 }}>
              searching...
            </div>
          )}
        </div>
      </Panel>

      {/* Dataset picker */}
      <Panel label="Dataset" style={{ padding: 14 }}>
        <div style={fieldStyle}>
          <label style={labelStyle}>JSONL Path</label>
          <div style={{ display: "flex", gap: 6 }}>
            <input
              style={{ ...inputStyle, flex: 1 }}
              value={config.dataset_path || ""}
              placeholder="/path/to/data.jsonl"
              onChange={(e) => setConfig((c) => ({ ...c, dataset_path: e.target.value }))}
            />
            <button
              onClick={handlePreviewDataset}
              disabled={!config.dataset_path || datasetPreviewing}
              style={{ padding: "6px 12px", whiteSpace: "nowrap", fontSize: "9px" }}
            >
              {datasetPreviewing ? "..." : "PREVIEW"}
            </button>
          </div>
        </div>

        {datasetPreview && (
          <div
            style={{
              marginTop: 8,
              background: "var(--grid)",
              padding: 8,
              fontSize: "10px",
              fontFamily: "'JetBrains Mono', monospace",
            }}
          >
            <div style={{ color: "var(--lime)", marginBottom: 4 }}>
              format: {datasetPreview.format}
            </div>
            {datasetPreview.rows.slice(0, 2).map((row, i) => (
              <div
                key={i}
                style={{
                  color: "var(--dim)",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                  fontSize: "9px",
                  borderTop: "1px solid var(--border)",
                  paddingTop: 4,
                  marginTop: 4,
                }}
              >
                {JSON.stringify(row).slice(0, 80)}...
              </div>
            ))}
          </div>
        )}

        <div style={{ ...fieldStyle, marginTop: 12 }}>
          <label style={labelStyle}>Output Directory</label>
          <input
            style={inputStyle}
            value={config.output_dir || ""}
            placeholder="/path/to/outputs"
            onChange={(e) => setConfig((c) => ({ ...c, output_dir: e.target.value }))}
          />
        </div>
      </Panel>

      {/* Method selector */}
      <Panel label="Method" style={{ padding: 14 }}>
        <div className="seg-ctrl">
          {METHODS.map((m) => (
            <button
              key={m}
              className={config.finetuning_method === m ? "active" : ""}
              onClick={() => setConfig((c) => ({ ...c, finetuning_method: m as Method }))}
              style={{ fontSize: "9px", padding: "5px 8px" }}
            >
              {m.toUpperCase()}
            </button>
          ))}
        </div>
      </Panel>

      {/* Hyperparameters */}
      <Panel label="Hyperparameters" style={{ padding: 14 }}>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: "10px 14px",
          }}
        >
          {[
            { key: "batch_size", label: "Batch Size", type: "number", min: 1 },
            { key: "gradient_accumulation_steps", label: "Grad Accum", type: "number", min: 1 },
            { key: "learning_rate", label: "Learning Rate", type: "number", step: "0.00001" },
            { key: "num_epochs", label: "Epochs", type: "number", min: 1 },
            { key: "max_seq_length", label: "Max Seq Length", type: "number", min: 64 },
          ].map(({ key, label, type: _t, ...attrs }) => (
            <div key={key}>
              <label style={labelStyle}>{label}</label>
              <input
                style={inputStyle}
                type="number"
                value={(config as any)[key] ?? ""}
                {...attrs}
                onChange={(e) =>
                  setConfig((c) => ({
                    ...c,
                    [key]: parseFloat(e.target.value),
                  }))
                }
              />
            </div>
          ))}
        </div>

        {/* LoRA params (shown when method is lora/qlora) */}
        {(config.finetuning_method === "lora" || config.finetuning_method === "qlora") && (
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr 1fr",
              gap: "10px 14px",
              marginTop: 12,
              paddingTop: 10,
              borderTop: "1px solid var(--border)",
            }}
          >
            {[
              { key: "lora_r", label: "LoRA r" },
              { key: "lora_alpha", label: "LoRA alpha" },
              { key: "lora_dropout", label: "Dropout" },
            ].map(({ key, label }) => (
              <div key={key}>
                <label style={labelStyle}>{label}</label>
                <input
                  style={inputStyle}
                  type="number"
                  value={(config as any)[key] ?? ""}
                  onChange={(e) =>
                    setConfig((c) => ({ ...c, [key]: parseFloat(e.target.value) }))
                  }
                />
              </div>
            ))}
          </div>
        )}

        {/* Toggles */}
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: "8px 16px",
            marginTop: 12,
            paddingTop: 10,
            borderTop: "1px solid var(--border)",
          }}
        >
          {[
            { key: "pack_sequences", label: "Pack Sequences" },
            { key: "stream", label: "Stream Layers" },
            { key: "compile", label: "Torch Compile" },
            { key: "zero_offload", label: "Zero Offload" },
          ].map(({ key, label }) => (
            <label
              key={key}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 6,
                cursor: "pointer",
                fontSize: "10px",
                fontFamily: "'Share Tech Mono', monospace",
                letterSpacing: "1px",
                color: "var(--dim)",
              }}
            >
              <div
                style={{
                  width: 14,
                  height: 14,
                  border: `1px solid ${(config as any)[key] ? "var(--cyan)" : "var(--border)"}`,
                  background: (config as any)[key] ? "rgba(0,229,255,0.2)" : "transparent",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  cursor: "pointer",
                  transition: "all 0.15s",
                }}
                onClick={() => setConfig((c) => ({ ...c, [key]: !(c as any)[key] }))}
              >
                {(config as any)[key] && (
                  <div
                    style={{ width: 6, height: 6, background: "var(--cyan)" }}
                  />
                )}
              </div>
              {label}
            </label>
          ))}
        </div>
      </Panel>

      {/* VRAM estimate + Start */}
      <Panel style={{ padding: 14 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div>
            <div style={{ fontSize: "9px", letterSpacing: "2px", textTransform: "uppercase", color: "var(--dim)", fontFamily: "'Share Tech Mono', monospace", marginBottom: 4 }}>
              VRAM Estimate
            </div>
            <div
              className="num"
              style={{
                fontSize: "22px",
                fontFamily: "'JetBrains Mono', monospace",
                color: vramColor,
              }}
            >
              ~{vramEst} GB
              <span style={{ fontSize: "10px", color: "var(--dim)", marginLeft: 4 }}>
                / 24 GB
              </span>
            </div>
          </div>

          <button
            onClick={handleStart}
            disabled={starting || !config.model_name_or_path || !config.dataset_path}
            style={{
              fontSize: "11px",
              padding: "10px 28px",
              background: starting ? "transparent" : "rgba(0,229,255,0.1)",
              borderColor: "var(--cyan)",
              color: "var(--cyan)",
              letterSpacing: "3px",
            }}
          >
            {starting ? "LAUNCHING..." : "START TRAINING"}
          </button>
        </div>

        {error && (
          <div
            style={{
              marginTop: 10,
              padding: "6px 10px",
              background: "rgba(255,43,214,0.1)",
              border: "1px solid var(--magenta)",
              color: "var(--magenta)",
              fontSize: "11px",
            }}
          >
            {error}
          </div>
        )}
      </Panel>
    </div>
  );
}
