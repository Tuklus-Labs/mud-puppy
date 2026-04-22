/**
 * Logs pane — virtualized stderr tail from sidecar.
 *
 * Features:
 * - Virtualized list (100px visible window using CSS)
 * - Filter box (substring)
 * - Level chips: info / warn / error
 * - Copy-all button
 */
import React, { useRef, useEffect, useState, useMemo } from "react";
import { useStore } from "../lib/store";
import type { LogEntry } from "../lib/store";
import { Panel } from "../chrome/VectorFrame";

const LEVEL_COLORS: Record<LogEntry["level"], string> = {
  info: "var(--dim)",
  warn: "var(--amber)",
  error: "var(--magenta)",
};

const LEVEL_LABELS: LogEntry["level"][] = ["info", "warn", "error"];

export function Logs() {
  const logs = useStore((s) => s.logs);
  const logFilter = useStore((s) => s.logFilter);
  const setLogFilter = useStore((s) => s.setLogFilter);
  const activeRunId = useStore((s) => s.activeRunId);

  const [levelFilter, setLevelFilter] = useState<Set<LogEntry["level"]>>(
    new Set(["info", "warn", "error"])
  );
  const [autoScroll, setAutoScroll] = useState(true);
  const listRef = useRef<HTMLDivElement>(null);

  const filteredLogs = useMemo(() => {
    return logs.filter((entry) => {
      if (!levelFilter.has(entry.level)) return false;
      if (activeRunId && entry.run_id && entry.run_id !== activeRunId) return false;
      if (logFilter && !entry.line.toLowerCase().includes(logFilter.toLowerCase())) return false;
      return true;
    });
  }, [logs, logFilter, levelFilter, activeRunId]);

  // Auto-scroll to bottom
  useEffect(() => {
    if (autoScroll && listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
  }, [filteredLogs, autoScroll]);

  const handleScroll = () => {
    if (!listRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = listRef.current;
    const atBottom = scrollHeight - scrollTop - clientHeight < 40;
    setAutoScroll(atBottom);
  };

  const handleCopyAll = () => {
    const text = filteredLogs.map((e) => e.line).join("\n");
    navigator.clipboard.writeText(text).catch(() => {
      // Fallback for non-HTTPS
      const ta = document.createElement("textarea");
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
    });
  };

  const toggleLevel = (level: LogEntry["level"]) => {
    setLevelFilter((prev) => {
      const next = new Set(prev);
      if (next.has(level)) {
        if (next.size > 1) next.delete(level); // Keep at least one
      } else {
        next.add(level);
      }
      return next;
    });
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        padding: "8px 12px",
      }}
    >
      {/* Toolbar */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          marginBottom: 8,
          flexShrink: 0,
        }}
      >
        {/* Filter input */}
        <input
          style={{
            flex: 1,
            background: "var(--panel-hi)",
            border: "1px solid var(--border)",
            color: "var(--text)",
            padding: "5px 10px",
            fontSize: "11px",
            fontFamily: "'JetBrains Mono', monospace",
            outline: "none",
          }}
          placeholder="filter..."
          value={logFilter}
          onChange={(e) => setLogFilter(e.target.value)}
        />

        {/* Level chips */}
        <div style={{ display: "flex", gap: 4 }}>
          {LEVEL_LABELS.map((level) => (
            <button
              key={level}
              onClick={() => toggleLevel(level)}
              style={{
                fontSize: "9px",
                padding: "3px 8px",
                borderColor: levelFilter.has(level) ? LEVEL_COLORS[level] : "var(--border)",
                color: levelFilter.has(level) ? LEVEL_COLORS[level] : "var(--dim)",
                background: levelFilter.has(level)
                  ? `${LEVEL_COLORS[level]}22`
                  : "transparent",
              }}
            >
              {level.toUpperCase()}
            </button>
          ))}
        </div>

        {/* Count */}
        <span
          style={{
            fontSize: "10px",
            fontFamily: "'JetBrains Mono', monospace",
            color: "var(--dim)",
            minWidth: 60,
            textAlign: "right",
          }}
        >
          {filteredLogs.length} lines
        </span>

        {/* Copy button */}
        <button
          onClick={handleCopyAll}
          style={{ fontSize: "9px", padding: "3px 10px" }}
        >
          COPY
        </button>

        {/* Auto-scroll indicator */}
        <button
          onClick={() => {
            setAutoScroll(true);
            if (listRef.current) {
              listRef.current.scrollTop = listRef.current.scrollHeight;
            }
          }}
          style={{
            fontSize: "9px",
            padding: "3px 8px",
            borderColor: autoScroll ? "var(--lime)" : "var(--border)",
            color: autoScroll ? "var(--lime)" : "var(--dim)",
          }}
        >
          {autoScroll ? "LIVE" : "TAIL"}
        </button>
      </div>

      {/* Log list */}
      <Panel
        style={{
          flex: 1,
          overflow: "hidden",
          display: "flex",
          flexDirection: "column",
        }}
      >
        <div
          ref={listRef}
          onScroll={handleScroll}
          style={{
            flex: 1,
            overflowY: "auto",
            padding: "6px 0",
          }}
        >
          {filteredLogs.length === 0 ? (
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                padding: 40,
                color: "var(--dim)",
                fontFamily: "'Share Tech Mono', monospace",
                fontSize: "11px",
                letterSpacing: "2px",
              }}
            >
              {logs.length === 0 ? "NO LOGS YET" : "NO MATCHING LOGS"}
            </div>
          ) : (
            filteredLogs.map((entry) => (
              <div
                key={entry.id}
                style={{
                  display: "flex",
                  gap: 8,
                  padding: "1px 12px",
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: "11px",
                  lineHeight: 1.5,
                  borderLeft: entry.level !== "info"
                    ? `2px solid ${LEVEL_COLORS[entry.level]}`
                    : "2px solid transparent",
                }}
              >
                {/* Level indicator */}
                {entry.level !== "info" && (
                  <span
                    style={{
                      color: LEVEL_COLORS[entry.level],
                      flexShrink: 0,
                      fontSize: "9px",
                      textTransform: "uppercase",
                      letterSpacing: "1px",
                      paddingTop: 2,
                    }}
                  >
                    {entry.level}
                  </span>
                )}
                {/* Line content */}
                <span
                  style={{
                    color: LEVEL_COLORS[entry.level],
                    wordBreak: "break-all",
                    flex: 1,
                  }}
                >
                  {entry.line}
                </span>
              </div>
            ))
          )}
        </div>
      </Panel>
    </div>
  );
}
