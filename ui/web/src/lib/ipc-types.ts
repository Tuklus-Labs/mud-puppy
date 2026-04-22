/**
 * IPC type definitions for mud-puppy-studio.
 *
 * Hand-written to match the manifest.toml IPC surface.
 * The C workstream will regenerate this via `phos types` — regeneration
 * must be idempotent with this version (types are conservative and precise).
 *
 * DO NOT add optional fields that don't exist in the manifest.
 * DO NOT widen union types.
 */

// ---------------------------------------------------------------------------
// Core training types
// ---------------------------------------------------------------------------

export interface TrainingConfig {
  model_name_or_path: string;
  dataset_path: string;
  output_dir: string;
  finetuning_method: "full" | "lora" | "qlora" | "gptq" | "qat" | "preference" | "rl" | "multimodal" | "rm" | "prm" | "embedding";
  /** Sub-type for preference methods (dpo, ipo, kto, orpo). Only used when finetuning_method is "preference". */
  preference?: "dpo" | "ipo" | "kto" | "orpo";
  num_epochs?: number;
  batch_size?: number;
  gradient_accumulation?: number;
  learning_rate?: number;
  max_seq_length?: number;
  lora_r?: number;
  lora_alpha?: number;
  lora_dropout?: number;
  pack_sequences?: boolean;
  stream?: boolean;
  prefetch_layers?: number;
  compile?: boolean;
  compile_mode?: "default" | "reduce-overhead" | "max-autotune";
  zero_offload?: boolean;
  monitor?: boolean;
  monitor_port?: number;
}

export interface RunHandle {
  run_id: string;
  /** pid is informational only; not used for monitor connection (async via port announcement). */
  pid?: number;
}

export interface RunSummary {
  run_id: string;
  model: string;
  method: string;
  dataset: string;
  status: "running" | "stopping" | "complete" | "failed" | "stopped";
  start_time: number;
  end_time?: number;
  final_loss?: number;
  steps_total?: number;
  steps_done?: number;
}

// ---------------------------------------------------------------------------
// HuggingFace model search
// ---------------------------------------------------------------------------

export interface HFModel {
  id: string;
  downloads: number;
  likes: number;
  tags: string[];
}

// ---------------------------------------------------------------------------
// Checkpoints
// ---------------------------------------------------------------------------

export interface Checkpoint {
  path: string;
  step?: number;
  loss?: number;
  /**
   * Unix epoch seconds (NOT milliseconds). Multiply by 1000 before
   * feeding to JS Date. The C workstream must preserve this unit when
   * regenerating types via `phos types`.
   */
  save_time_s?: number;
  is_lora: boolean;
}

// ---------------------------------------------------------------------------
// IPC events
// ---------------------------------------------------------------------------

export interface MetricsEvent {
  type: "metrics";
  run_id: string;
  step: number;
  loss: number;
  learning_rate?: number;
  grad_norm?: number;
  tokens_per_sec?: number;
  steps_per_sec?: number;
  epoch?: number;
  lora_norms?: Record<string, number[]>;
}

export interface GpuEvent {
  type: "gpu";
  gpu_util_pct: number;
  vram_used_gb: number;
  vram_total_gb: number;
  temp_c?: number;
  power_w?: number;
}

export interface StreamStatsEvent {
  type: "stream_stats";
  layers_total: number;
  layers_resident: number;
  h2d_bandwidth_gbps: number;
  prefetch_hit_rate: number;
  layer_states?: Array<{
    idx: number;
    resident: boolean;
    prefetching: boolean;
    last_h2d_ms?: number;
  }>;
}

export interface MemoryStatsEvent {
  type: "memory_stats";
  allocated_gb: number;
  reserved_gb: number;
  active_gb: number;
  fragmentation: number;
  model_gb?: number;
  optimizer_gb?: number;
  activations_gb?: number;
  kv_gb?: number;
}

export interface LogLineEvent {
  type: "log_line";
  run_id: string;
  line: string;
}

export interface RunCompleteEvent {
  type: "run_complete";
  run_id: string;
  exit_code: number;
}

export type AnyEvent =
  | MetricsEvent
  | GpuEvent
  | StreamStatsEvent
  | MemoryStatsEvent
  | LogLineEvent
  | RunCompleteEvent;
