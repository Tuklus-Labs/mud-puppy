/**
 * Typed wrappers around window.phos.invoke / window.phos.on.
 *
 * In dev mode (no Phos host), window.phos is stubbed with mock data
 * so all five panes render without crashing.
 */
import type {
  TrainingConfig,
  RunSummary,
  HFModel,
  Checkpoint,
  MetricsEvent,
  GpuEvent,
  StreamStatsEvent,
  MemoryStatsEvent,
  LogLineEvent,
  RunCompleteEvent,
} from "./ipc-types";

// ---------------------------------------------------------------------------
// Phos bridge interface
// ---------------------------------------------------------------------------

interface PhosBridge {
  invoke<T = unknown>(name: string, args?: unknown): Promise<T>;
  on(event: string, cb: (payload: unknown) => void): () => void;
  off?(event: string, cb: (payload: unknown) => void): void;
}

declare global {
  interface Window {
    phos?: PhosBridge;
  }
}

// ---------------------------------------------------------------------------
// Dev-mode stub — renders mock data when no Phos host is present
// ---------------------------------------------------------------------------

const MOCK_RUNS: RunSummary[] = [
  {
    run_id: "mock-run-001",
    model: "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    method: "lora",
    dataset: "data/sample.jsonl",
    status: "complete",
    start_time: Date.now() - 3600000,
    end_time: Date.now() - 1800000,
    final_loss: 1.243,
    steps_total: 500,
    steps_done: 500,
  },
  {
    run_id: "mock-run-002",
    model: "mistralai/Mistral-7B-v0.1",
    method: "qlora",
    dataset: "data/chat.jsonl",
    status: "complete",
    start_time: Date.now() - 7200000,
    end_time: Date.now() - 5400000,
    final_loss: 0.987,
    steps_total: 1000,
    steps_done: 1000,
  },
];

const MOCK_MODELS: HFModel[] = [
  { id: "meta-llama/Llama-3-8B", downloads: 1200000, likes: 4500, tags: ["llama", "causal-lm"] },
  { id: "mistralai/Mistral-7B-v0.1", downloads: 980000, likes: 3200, tags: ["mistral", "causal-lm"] },
  { id: "TinyLlama/TinyLlama-1.1B-Chat-v1.0", downloads: 650000, likes: 2100, tags: ["tinyllama", "chat"] },
  { id: "Qwen/Qwen2.5-7B", downloads: 500000, likes: 1800, tags: ["qwen", "causal-lm"] },
];

const mockPhosEventHandlers: Map<string, Set<(p: unknown) => void>> = new Map();

const mockPhos: PhosBridge = {
  async invoke<T>(name: string, _args?: unknown): Promise<T> {
    switch (name) {
      case "run.list":
        return MOCK_RUNS as unknown as T;
      case "run.start":
        return { run_id: `mock-run-${Date.now()}`, port: 5980 } as unknown as T;
      case "run.stop":
        return { ok: true } as unknown as T;
      case "hf.search_models":
        return MOCK_MODELS as unknown as T;
      case "checkpoint.list":
        return [] as unknown as T;
      case "dataset.preview":
        return {
          format: "messages",
          rows: [
            { messages: [{ role: "user", content: "Hello" }, { role: "assistant", content: "Hi!" }] },
            { messages: [{ role: "user", content: "Tell me a story." }, { role: "assistant", content: "Once upon a time..." }] },
          ],
        } as unknown as T;
      default:
        return null as unknown as T;
    }
  },
  on(event: string, cb: (payload: unknown) => void): () => void {
    if (!mockPhosEventHandlers.has(event)) {
      mockPhosEventHandlers.set(event, new Set());
    }
    mockPhosEventHandlers.get(event)!.add(cb);
    return () => mockPhosEventHandlers.get(event)?.delete(cb);
  },
};

// ---------------------------------------------------------------------------
// Bridge resolution
// ---------------------------------------------------------------------------

function getBridge(): PhosBridge {
  if (typeof window !== "undefined" && window.phos) {
    return window.phos;
  }
  // Dev stub
  return mockPhos;
}

// ---------------------------------------------------------------------------
// Typed IPC surface
// ---------------------------------------------------------------------------

export const ipc = {
  // Commands
  startRun: (config: TrainingConfig) =>
    getBridge().invoke<{ run_id: string; port: number }>("run.start", { config }),

  stopRun: (run_id: string) =>
    getBridge().invoke<{ ok: boolean }>("run.stop", { run_id }),

  listRuns: () => {
    const result = getBridge().invoke<RunSummary[] | null>("run.list");
    return result.then((r) => (Array.isArray(r) ? r : []));
  },

  searchModels: (query: string) => {
    const result = getBridge().invoke<HFModel[] | null>("hf.search_models", { query });
    return result.then((r) => (Array.isArray(r) ? r : []));
  },

  listCheckpoints: (output_dir: string) => {
    const result = getBridge().invoke<Checkpoint[] | null>("checkpoint.list", { output_dir });
    return result.then((r) => (Array.isArray(r) ? r : []));
  },

  previewDataset: (path: string, n = 5) =>
    getBridge().invoke<{ format: string; rows: unknown[] }>("dataset.preview", { path, n }),

  // Events
  onMetrics: (cb: (m: MetricsEvent) => void) =>
    getBridge().on("metrics", (p) => cb(p as MetricsEvent)),

  onGpu: (cb: (m: GpuEvent) => void) =>
    getBridge().on("gpu", (p) => cb(p as GpuEvent)),

  onStreamStats: (cb: (m: StreamStatsEvent) => void) =>
    getBridge().on("stream_stats", (p) => cb(p as StreamStatsEvent)),

  onMemoryStats: (cb: (m: MemoryStatsEvent) => void) =>
    getBridge().on("memory_stats", (p) => cb(p as MemoryStatsEvent)),

  onLogLine: (cb: (m: LogLineEvent) => void) =>
    getBridge().on("log_line", (p) => cb(p as LogLineEvent)),

  onRunComplete: (cb: (m: RunCompleteEvent) => void) =>
    getBridge().on("run_complete", (p) => cb(p as RunCompleteEvent)),
};

// Export mock emitter for dev mode testing
export function devEmit(event: string, payload: unknown) {
  const handlers = mockPhosEventHandlers.get(event);
  if (handlers) {
    handlers.forEach((h) => h(payload));
  }
}
