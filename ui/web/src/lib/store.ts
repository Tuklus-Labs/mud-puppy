/**
 * Zustand global store for mud-puppy-studio.
 * All IPC event consumers write here; all panels read from here.
 */
import { create } from "zustand";
import type {
  RunSummary,
  MetricsEvent,
  GpuEvent,
  StreamStatsEvent,
  MemoryStatsEvent,
  LogLineEvent,
} from "./ipc-types";

// Ring buffer helper — single immutable slice, no in-place mutation after spread.
function ringPush<T>(arr: T[], item: T, max: number): T[] {
  const next = [...arr, item];
  return next.length > max ? next.slice(next.length - max) : next;
}

export interface LogEntry {
  id: number;
  run_id: string;
  line: string;
  level: "info" | "warn" | "error";
  timestamp: number;
}

let _logId = 0;

function classifyLevel(line: string): LogEntry["level"] {
  const l = line.toLowerCase();
  if (l.includes("error") || l.includes("traceback") || l.includes("exception")) return "error";
  if (l.includes("warn") || l.includes("warning")) return "warn";
  return "info";
}

// ---------------------------------------------------------------------------
// Store shape
// ---------------------------------------------------------------------------

interface StudioState {
  // Navigation
  activePane: "launch" | "monitor" | "runs" | "library" | "logs";
  setActivePane: (p: StudioState["activePane"]) => void;

  // Active run
  activeRunId: string | null;
  setActiveRunId: (id: string | null) => void;

  // Run list
  runs: RunSummary[];
  setRuns: (runs: RunSummary[]) => void;
  upsertRun: (run: RunSummary) => void;

  // Compare mode (two-run overlay on loss chart)
  compareRunId: string | null;
  setCompareRunId: (id: string | null) => void;

  // Metrics history — ring buffer 10k per run
  metricsHistory: Record<string, MetricsEvent[]>;
  appendMetrics: (m: MetricsEvent) => void;

  // GPU telemetry — ring buffer 300 points (5 min @ 1s)
  gpuHistory: GpuEvent[];
  appendGpu: (g: GpuEvent) => void;

  // Latest streaming stats
  streamStats: StreamStatsEvent | null;
  setStreamStats: (s: StreamStatsEvent) => void;

  // Latest memory stats
  memoryStats: MemoryStatsEvent | null;
  setMemoryStats: (s: MemoryStatsEvent) => void;

  // Logs — ring buffer 10k lines
  logs: LogEntry[];
  appendLog: (e: LogLineEvent) => void;
  logFilter: string;
  setLogFilter: (f: string) => void;

  // Background shader toggle
  backgroundEnabled: boolean;
  toggleBackground: () => void;

  // Wire flash state — which event name was last seen (for VectorWires)
  lastEvent: string | null;
  setLastEvent: (name: string | null) => void;
}

export const useStore = create<StudioState>()((set, get) => ({
  // Navigation
  activePane: "launch",
  setActivePane: (p) => set({ activePane: p }),

  // Active run
  activeRunId: null,
  setActiveRunId: (id) => set({ activeRunId: id }),

  // Run list
  runs: [],
  setRuns: (runs) => set({ runs }),
  upsertRun: (run) => {
    const current = get().runs;
    const idx = current.findIndex((r) => r.run_id === run.run_id);
    if (idx >= 0) {
      const updated = [...current];
      updated[idx] = run;
      set({ runs: updated });
    } else {
      set({ runs: [run, ...current] });
    }
  },

  // Compare
  compareRunId: null,
  setCompareRunId: (id) => set({ compareRunId: id }),

  // Metrics
  metricsHistory: {},
  appendMetrics: (m) => {
    const history = get().metricsHistory;
    const existing = history[m.run_id] || [];
    set({
      metricsHistory: {
        ...history,
        [m.run_id]: ringPush(existing, m, 10000),
      },
      lastEvent: "metrics",
    });
  },

  // GPU
  gpuHistory: [],
  appendGpu: (g) => {
    set({ gpuHistory: ringPush(get().gpuHistory, g, 300), lastEvent: "gpu" });
  },

  // Stream stats
  streamStats: null,
  setStreamStats: (s) => set({ streamStats: s, lastEvent: "stream_stats" }),

  // Memory stats
  memoryStats: null,
  setMemoryStats: (s) => set({ memoryStats: s, lastEvent: "memory_stats" }),

  // Logs
  logs: [],
  appendLog: (e) => {
    const entry: LogEntry = {
      id: ++_logId,
      run_id: e.run_id,
      line: e.line,
      level: classifyLevel(e.line),
      timestamp: Date.now(),
    };
    set({ logs: ringPush(get().logs, entry, 10000), lastEvent: "log_line" });
  },
  logFilter: "",
  setLogFilter: (f) => set({ logFilter: f }),

  // Background
  backgroundEnabled: (() => {
    if (typeof window === "undefined") return true;
    if (new URLSearchParams(window.location.search).get("chrome") === "minimal") return false;
    const saved = localStorage.getItem("mp-bg-enabled");
    if (saved !== null) return saved === "true";
    // No saved preference: honor prefers-reduced-motion (default off if reduced).
    const reduced =
      window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches === true;
    return !reduced;
  })(),
  toggleBackground: () => {
    const next = !get().backgroundEnabled;
    localStorage.setItem("mp-bg-enabled", String(next));
    set({ backgroundEnabled: next });
  },

  // Wire flash
  lastEvent: null,
  setLastEvent: (name) => set({ lastEvent: name }),
}));
