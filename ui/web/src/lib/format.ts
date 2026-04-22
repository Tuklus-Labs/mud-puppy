/**
 * Formatting utilities for numeric display.
 */

export function fmtNum(n: number, decimals = 2): string {
  if (!isFinite(n)) return "--";
  return n.toFixed(decimals);
}

export function fmtGb(bytes: number): string {
  return (bytes / 1e9).toFixed(2) + " GB";
}

export function fmtGbps(gbps: number): string {
  if (gbps >= 100) return gbps.toFixed(0) + " GB/s";
  return gbps.toFixed(1) + " GB/s";
}

export function fmtPct(ratio: number): string {
  return (ratio * 100).toFixed(1) + "%";
}

export function fmtSec(ms: number): string {
  const s = ms / 1000;
  if (s < 60) return s.toFixed(1) + "s";
  const m = Math.floor(s / 60);
  const rem = Math.floor(s % 60);
  return `${m}m ${rem}s`;
}

export function fmtDuration(startMs: number, endMs?: number): string {
  const delta = (endMs ?? Date.now()) - startMs;
  return fmtSec(delta);
}

export function fmtRelTime(ms: number): string {
  const delta = Date.now() - ms;
  if (delta < 60000) return "just now";
  if (delta < 3600000) return Math.floor(delta / 60000) + "m ago";
  if (delta < 86400000) return Math.floor(delta / 3600000) + "h ago";
  return Math.floor(delta / 86400000) + "d ago";
}

export function fmtTokPerSec(n: number): string {
  if (n >= 1000) return (n / 1000).toFixed(1) + "k tok/s";
  return n.toFixed(0) + " tok/s";
}

export function fmtLoss(n: number): string {
  if (!isFinite(n)) return "--";
  return n.toFixed(4);
}

// Coefficient of variation color coding for throughput panels
export function cvColor(values: number[]): string {
  if (values.length < 2) return "var(--lime)";
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  if (mean === 0) return "var(--dim)";
  const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length;
  const cv = Math.sqrt(variance) / mean;
  if (cv <= 0.1) return "var(--lime)";
  if (cv <= 0.25) return "var(--amber)";
  return "var(--magenta)";
}
