/**
 * Launch pane — model / dataset / method / hyperparameters with a sticky
 * side column showing live VRAM estimate and run summary.
 *
 * Layout: .launch-grid (1fr + 300px, stacks <1100px).
 * Mirrors the 2026-04-22 design handoff pixel-for-pixel.
 */
import React, { useEffect, useMemo, useState } from "react";
import { Panel } from "../chrome/VectorFrame";
import { ipc } from "../lib/ipc";
import type { HFModel, TrainingConfig } from "../lib/ipc-types";
import { useStore } from "../lib/store";

const METHODS: TrainingConfig["finetuning_method"][] = [
  "lora",
  "qlora",
  "full",
  "dpo",
  "grpo",
  "orpo",
  "kto",
];

interface LaunchState {
  model: string;
  dataset: string;
  output: string;
  method: TrainingConfig["finetuning_method"];
  batch_size: number;
  grad_accum: number;
  learning_rate: number;
  num_epochs: number;
  max_seq_length: number;
  lora_r: number;
  lora_alpha: number;
  lora_dropout: number;
  pack: boolean;
  stream: boolean;
  compile: boolean;
  zero_offload: boolean;
  warmup_ratio: number;
}

const DEFAULT: LaunchState = {
  model: "meta-llama/Llama-3-8B-Instruct",
  dataset: "/data/instruct/alpaca-clean.jsonl",
  output: "outputs/llama3-finetune",
  method: "lora",
  batch_size: 4,
  grad_accum: 4,
  learning_rate: 2e-4,
  num_epochs: 3,
  max_seq_length: 2048,
  lora_r: 16,
  lora_alpha: 32,
  lora_dropout: 0.05,
  pack: true,
  stream: false,
  compile: false,
  zero_offload: false,
  warmup_ratio: 0.03,
};

function guessParams(model: string): number {
  const mn = model.toLowerCase();
  if (mn.includes("1b") || mn.includes("tiny")) return 1.1;
  if (mn.includes("3b")) return 3;
  if (mn.includes("8b") || mn.includes("9b")) return 8;
  if (mn.includes("13b")) return 13;
  if (mn.includes("70b")) return 70;
  if (mn.includes("7b") || mn.includes("mistral")) return 7;
  return 7;
}

function estimateVram(s: LaunchState) {
  const params = guessParams(s.model);
  const dtypeBytes = s.method === "qlora" ? 0.5 : 2;
  const modelGb = params * dtypeBytes;
  const optGb = s.method === "full" ? modelGb * 2.2 : modelGb * 0.15;
  const actGb =
    ((s.max_seq_length * s.batch_size * 4096 * 4) / 1e9) *
    (s.method === "qlora" ? 0.6 : 1);
  const total = modelGb + optGb + actGb;
  return {
    model: Math.round(modelGb * 10) / 10,
    opt: Math.round(optGb * 10) / 10,
    act: Math.round(actGb * 10) / 10,
    total: Math.round(total * 10) / 10,
  };
}

export function Launch() {
  const [cfg, setCfg] = useState<LaunchState>(DEFAULT);
  const [showDrop, setShowDrop] = useState(false);
  const [hfResults, setHfResults] = useState<HFModel[]>([]);
  const setActivePane = useStore((s) => s.setActivePane);

  const update = <K extends keyof LaunchState>(k: K, v: LaunchState[K]) =>
    setCfg((c) => ({ ...c, [k]: v }));

  useEffect(() => {
    if (!cfg.model || cfg.model.length < 3) {
      setHfResults([]);
      return;
    }
    const h = setTimeout(async () => {
      try {
        const r = await ipc.searchModels(cfg.model);
        setHfResults(Array.isArray(r) ? r.slice(0, 6) : []);
      } catch {
        setHfResults([]);
      }
    }, 260);
    return () => clearTimeout(h);
  }, [cfg.model]);

  const vram = useMemo(() => estimateVram(cfg), [cfg]);
  const budgetGb = 24;
  const pct = Math.min(100, (vram.total / budgetGb) * 100);
  const level = vram.total > 22 ? "danger" : vram.total > 16 ? "warn" : "";
  const vramColor =
    level === "danger"
      ? "var(--magenta)"
      : level === "warn"
      ? "var(--amber)"
      : "var(--lime)";

  const start = async () => {
    const config: TrainingConfig = {
      model_name_or_path: cfg.model,
      dataset_path: cfg.dataset,
      output_dir: cfg.output,
      finetuning_method: cfg.method,
      num_epochs: cfg.num_epochs,
      batch_size: cfg.batch_size,
      gradient_accumulation_steps: cfg.grad_accum,
      learning_rate: cfg.learning_rate,
      max_seq_length: cfg.max_seq_length,
      lora_r: cfg.lora_r,
      lora_alpha: cfg.lora_alpha,
      lora_dropout: cfg.lora_dropout,
      pack_sequences: cfg.pack,
      stream: cfg.stream,
      compile: cfg.compile,
      zero_offload: cfg.zero_offload,
      monitor: true,
    };
    try {
      const handle = await ipc.startRun(config);
      if (handle?.run_id) {
        useStore.getState().setActiveRunId(handle.run_id);
      }
      setActivePane("monitor");
    } catch (e) {
      console.error("[launch] startRun failed", e);
    }
  };

  const isLoraMethod = cfg.method === "lora" || cfg.method === "qlora";

  return (
    <div className="pane">
      <div className="pane-title">
        <h1>Launch</h1>
        <div className="crumb">
          MUD-PUPPY › <span>CONFIGURE RUN</span>
        </div>
      </div>

      <div className="launch-grid">
        <div className="launch-main">
          <Panel label="Model">
            <div className="field">
              <div className="label">Hugging Face ID or local path</div>
              <div style={{ position: "relative" }}>
                <input
                  value={cfg.model}
                  onChange={(e) => {
                    update("model", e.target.value);
                    setShowDrop(true);
                  }}
                  onFocus={() => setShowDrop(true)}
                  onBlur={() => setTimeout(() => setShowDrop(false), 140)}
                  placeholder="meta-llama/Llama-3-8B"
                />
                {showDrop && hfResults.length > 0 && (
                  <div className="dropdown">
                    {hfResults.map((r) => (
                      <div
                        key={r.id}
                        className="dropdown-row"
                        onMouseDown={() => {
                          update("model", r.id);
                          setShowDrop(false);
                        }}
                      >
                        <span>{r.id}</span>
                        <span className="meta">
                          {(r.downloads / 1e6).toFixed(1)}M ↓
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </Panel>

          <Panel label="Dataset">
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              <div className="field">
                <div className="label">JSONL file</div>
                <div className="row">
                  <input
                    value={cfg.dataset}
                    onChange={(e) => update("dataset", e.target.value)}
                    placeholder="/path/to/data.jsonl"
                  />
                  <button
                    className="btn btn-ghost"
                    onClick={async () => {
                      try {
                        const p = await ipc.previewDataset(cfg.dataset, 3);
                        console.log("[launch] preview", p);
                      } catch (e) {
                        console.error(e);
                      }
                    }}
                  >
                    Preview
                  </button>
                </div>
              </div>
              <div className="field">
                <div className="label">Output directory</div>
                <input
                  value={cfg.output}
                  onChange={(e) => update("output", e.target.value)}
                  placeholder="outputs/my-run"
                />
              </div>
            </div>
          </Panel>

          <Panel label="Method">
            <div className="seg" style={{ width: "100%", display: "flex" }}>
              {METHODS.map((m) => (
                <button
                  key={m}
                  className={cfg.method === m ? "active" : ""}
                  onClick={() => update("method", m)}
                  style={{ flex: 1 }}
                >
                  {m}
                </button>
              ))}
            </div>
          </Panel>

          <Panel label="Hyperparameters">
            <div className="grid-2" style={{ marginBottom: 16 }}>
              <NumField
                label="Batch size"
                value={cfg.batch_size}
                onChange={(v) => update("batch_size", v)}
              />
              <NumField
                label="Grad accum"
                value={cfg.grad_accum}
                onChange={(v) => update("grad_accum", v)}
              />
              <NumField
                label="Learning rate"
                value={cfg.learning_rate}
                step={0.00001}
                onChange={(v) => update("learning_rate", v)}
              />
              <NumField
                label="Epochs"
                value={cfg.num_epochs}
                onChange={(v) => update("num_epochs", v)}
              />
              <NumField
                label="Max seq length"
                value={cfg.max_seq_length}
                onChange={(v) => update("max_seq_length", v)}
              />
              <NumField
                label="Warmup ratio"
                value={cfg.warmup_ratio}
                step={0.01}
                onChange={(v) => update("warmup_ratio", v)}
              />
            </div>

            {isLoraMethod && (
              <>
                <div className="label" style={{ marginBottom: 8 }}>
                  LoRA configuration
                </div>
                <div className="grid-3" style={{ marginBottom: 16 }}>
                  <NumField
                    label="Rank (r)"
                    value={cfg.lora_r}
                    onChange={(v) => update("lora_r", v)}
                  />
                  <NumField
                    label="Alpha"
                    value={cfg.lora_alpha}
                    onChange={(v) => update("lora_alpha", v)}
                  />
                  <NumField
                    label="Dropout"
                    value={cfg.lora_dropout}
                    step={0.01}
                    onChange={(v) => update("lora_dropout", v)}
                  />
                </div>
              </>
            )}

            <div className="label" style={{ marginBottom: 10 }}>
              Optimizations
            </div>
            <div className="toggle-row">
              {(
                [
                  ["pack", "Pack sequences"],
                  ["stream", "Stream layers"],
                  ["compile", "Torch compile"],
                  ["zero_offload", "ZeRO offload"],
                ] as const
              ).map(([k, lbl]) => (
                <label
                  key={k}
                  className="toggle"
                  onClick={() => update(k, !cfg[k])}
                >
                  <span className={`chk ${cfg[k] ? "on" : ""}`}></span>
                  {lbl}
                </label>
              ))}
            </div>
          </Panel>
        </div>

        <div className="launch-side">
          <Panel label="VRAM Estimate" active>
            <div className="vram-card">
              <div>
                <div className="vram-num" style={{ color: vramColor }}>
                  {vram.total.toFixed(1)}
                  <small> / {budgetGb} GB</small>
                </div>
                <div style={{ marginTop: 10 }} className={`vram-bar ${level}`}>
                  <div style={{ width: `${pct}%` }} />
                </div>
              </div>

              <div className="breakdown">
                <div className="breakdown-row">
                  <span>Weights</span>
                  <b>{vram.model.toFixed(1)} GB</b>
                </div>
                <div className="breakdown-row">
                  <span>Optimizer</span>
                  <b>{vram.opt.toFixed(1)} GB</b>
                </div>
                <div className="breakdown-row">
                  <span>Activations</span>
                  <b>{vram.act.toFixed(1)} GB</b>
                </div>
                <div
                  className="breakdown-row"
                  style={{
                    marginTop: 6,
                    paddingTop: 6,
                    borderTop: "1px solid var(--border)",
                  }}
                >
                  <span style={{ color: "var(--text)" }}>Peak estimate</span>
                  <b style={{ color: vramColor }}>{vram.total.toFixed(1)} GB</b>
                </div>
              </div>

              <div
                className="help"
                style={{
                  fontSize: 10,
                  color: "var(--dim)",
                  fontFamily: "var(--font-num)",
                }}
              >
                Target device: AMD 7900 XTX · 24 GB
              </div>
            </div>
          </Panel>

          <Panel label="Summary">
            <div className="breakdown">
              <div className="breakdown-row">
                <span>Method</span>
                <b
                  style={{
                    color: "var(--cyan)",
                    textTransform: "uppercase",
                    fontFamily: "var(--font-heading)",
                    letterSpacing: "0.15em",
                  }}
                >
                  {cfg.method}
                </b>
              </div>
              <div className="breakdown-row">
                <span>Est. total steps</span>
                <b>
                  {Math.max(
                    1,
                    Math.round(
                      (12847 / Math.max(1, cfg.batch_size * cfg.grad_accum)) *
                        cfg.num_epochs,
                    ),
                  ).toLocaleString()}
                </b>
              </div>
              <div className="breakdown-row">
                <span>Trainable params</span>
                <b>
                  {cfg.method === "full"
                    ? `${guessParams(cfg.model).toFixed(1)} B`
                    : `${(cfg.lora_r * 4.2).toFixed(1)} M`}
                </b>
              </div>
            </div>
          </Panel>

          <button
            className="btn btn-primary"
            onClick={start}
            style={{ justifyContent: "center" }}
          >
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
              <polygon points="2,1 11,6 2,11" fill="currentColor" />
            </svg>
            Start training
          </button>
        </div>
      </div>
    </div>
  );
}

function NumField({
  label,
  value,
  onChange,
  step,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  step?: number;
}) {
  return (
    <div className="field">
      <div className="label">{label}</div>
      <input
        type="number"
        value={value}
        step={step}
        onChange={(e) => {
          const v = parseFloat(e.target.value);
          if (Number.isFinite(v)) onChange(v);
        }}
      />
    </div>
  );
}
