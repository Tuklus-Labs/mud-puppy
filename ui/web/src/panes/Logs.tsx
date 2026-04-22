/**
 * Logs pane — filter toolbar + level chips + virtualized row list.
 *
 * Grid: 88px timestamp · 52px level · 1fr message. Rows are fixed 20px tall
 * so windowed virtualization works from scrollTop/rowHeight. Virtualization
 * is critical at 10k lines or the Logs pane freezes the app when training
 * is hot.
 */
import React, { useEffect, useMemo, useRef, useState } from "react";
import { Panel } from "../chrome/VectorFrame";
import { useStore } from "../lib/store";

const ROW_HEIGHT = 20;
const OVERSCAN = 10;

export function Logs() {
  const logs = useStore((s) => s.logs);
  const logFilter = useStore((s) => s.logFilter);
  const setLogFilter = useStore((s) => s.setLogFilter);

  const [levels, setLevels] = useState({ info: true, warn: true, error: true });
  const [live, setLive] = useState(true);

  const filtered = useMemo(() => {
    const needle = logFilter.toLowerCase();
    return logs.filter(
      (l) =>
        levels[l.level as keyof typeof levels] &&
        (!needle || l.line.toLowerCase().includes(needle)),
    );
  }, [logs, logFilter, levels]);

  // ───── Windowed virtualization ─────
  const scrollRef = useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [viewportH, setViewportH] = useState(0);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const h = entries[0]?.contentRect.height ?? 0;
      setViewportH(h);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Auto-scroll to bottom in live mode when new entries arrive.
  useEffect(() => {
    if (!live || !scrollRef.current) return;
    const el = scrollRef.current;
    el.scrollTop = el.scrollHeight;
  }, [filtered.length, live]);

  const totalHeight = filtered.length * ROW_HEIGHT;
  const visibleCount = Math.ceil(viewportH / ROW_HEIGHT) + OVERSCAN;
  const firstVisible = Math.max(
    0,
    Math.floor(scrollTop / ROW_HEIGHT) - OVERSCAN,
  );
  const lastVisible = Math.min(
    filtered.length,
    firstVisible + visibleCount + OVERSCAN,
  );
  const window = filtered.slice(firstVisible, lastVisible);
  const topSpacer = firstVisible * ROW_HEIGHT;
  const bottomSpacer = (filtered.length - lastVisible) * ROW_HEIGHT;

  const copyAll = () => {
    const txt = filtered
      .map(
        (l) =>
          `${new Date(l.timestamp).toISOString().slice(11, 19)} ${l.level.toUpperCase()} ${l.line}`,
      )
      .join("\n");
    navigator.clipboard?.writeText(txt).catch(() => {});
  };

  return (
    <div className="pane">
      <div className="pane-title">
        <h1>Logs</h1>
        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <span
            className="mono"
            style={{ fontSize: "var(--f-micro)", color: "var(--dim)" }}
          >
            {filtered.length.toLocaleString()} / {logs.length.toLocaleString()}{" "}
            lines
          </span>
          <button className="btn btn-ghost" onClick={copyAll}>
            Copy all
          </button>
          <button
            className={`btn ${live ? "btn-lime" : "btn-ghost"}`}
            onClick={() => setLive((l) => !l)}
          >
            <span
              className="dot"
              style={{
                width: 6,
                height: 6,
                background: live ? "var(--lime)" : "var(--dim)",
                boxShadow: live ? "0 0 6px var(--lime)" : "none",
              }}
            />
            {live ? "Live" : "Paused"}
          </button>
        </div>
      </div>

      <div className="logs-toolbar">
        <input
          value={logFilter}
          onChange={(e) => setLogFilter(e.target.value)}
          placeholder="filter logs…"
          style={{ flex: 1 }}
        />
        <div className="seg">
          {(["info", "warn", "error"] as const).map((l) => (
            <button
              key={l}
              className={levels[l] ? "active" : ""}
              onClick={() => setLevels((s) => ({ ...s, [l]: !s[l] }))}
              style={{
                color: levels[l]
                  ? l === "warn"
                    ? "var(--amber)"
                    : l === "error"
                    ? "var(--magenta)"
                    : "var(--cyan)"
                  : undefined,
              }}
            >
              {l}
            </button>
          ))}
        </div>
      </div>

      <Panel bodyStyle={{ padding: "12px 0" }}>
        <div
          ref={scrollRef}
          onScroll={(e) => setScrollTop((e.target as HTMLDivElement).scrollTop)}
          style={{ height: "calc(100vh - 280px)", overflowY: "auto" }}
        >
          <div style={{ height: topSpacer }} />
          {window.map((l) => (
            <div key={l.id} className={`log-row ${l.level}`}>
              <span className="log-time">
                {new Date(l.timestamp).toISOString().slice(11, 19)}
              </span>
              <span className="log-level">{l.level}</span>
              <span className="log-msg">{l.line}</span>
            </div>
          ))}
          <div style={{ height: bottomSpacer }} />
          {filtered.length === 0 && (
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
              NO LOG LINES YET
            </div>
          )}
          {/* Keep totalHeight tracked in a hidden element so layout is stable
              on rapid resize */}
          <div style={{ position: "absolute", top: 0, left: 0, width: 0, height: totalHeight, pointerEvents: "none" }} />
        </div>
      </Panel>
    </div>
  );
}
