/**
 * App shell — sidebar navigation + active pane.
 *
 * Keyboard shortcuts:
 *   1-5: jump to panes (Launch, Monitor, Runs, Library, Logs)
 *   Ctrl+Shift+M: toggle WebGL background
 */
import React, { useEffect, useCallback } from "react";
import { useStore } from "./lib/store";
import { Background } from "./chrome/Background";
import { Launch } from "./panes/Launch";
import { Monitor } from "./panes/Monitor";
import { Runs } from "./panes/Runs";
import { Library } from "./panes/Library";
import { Logs } from "./panes/Logs";

type Pane = "launch" | "monitor" | "runs" | "library" | "logs";

const PANES: { id: Pane; label: string; key: string }[] = [
  { id: "launch",  label: "LAUNCH",  key: "1" },
  { id: "monitor", label: "MONITOR", key: "2" },
  { id: "runs",    label: "RUNS",    key: "3" },
  { id: "library", label: "LIBRARY", key: "4" },
  { id: "logs",    label: "LOGS",    key: "5" },
];

function PaneContent({ pane }: { pane: Pane }) {
  switch (pane) {
    case "launch":  return <Launch />;
    case "monitor": return <Monitor />;
    case "runs":    return <Runs />;
    case "library": return <Library />;
    case "logs":    return <Logs />;
    default:        return <Launch />;
  }
}

export function App() {
  const activePane = useStore((s) => s.activePane);
  const setActivePane = useStore((s) => s.setActivePane);
  const toggleBackground = useStore((s) => s.toggleBackground);
  const runs = useStore((s) => s.runs);
  const activeRunId = useStore((s) => s.activeRunId);

  const activeRun = activeRunId ? runs.find((r) => r.run_id === activeRunId) : null;
  const isRunning = activeRun?.status === "running";

  const handleKey = useCallback(
    (e: KeyboardEvent) => {
      // Ctrl+Shift+M: toggle background
      if (e.ctrlKey && e.shiftKey && e.key === "M") {
        e.preventDefault();
        toggleBackground();
        return;
      }
      // Number keys 1-5: switch panes
      if (!e.ctrlKey && !e.altKey && !e.metaKey) {
        const pane = PANES.find((p) => p.key === e.key);
        if (pane && !(e.target instanceof HTMLInputElement) && !(e.target instanceof HTMLTextAreaElement)) {
          e.preventDefault();
          setActivePane(pane.id);
        }
      }
    },
    [setActivePane, toggleBackground]
  );

  useEffect(() => {
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [handleKey]);

  return (
    <div
      className="scanlines"
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        width: "100%",
        position: "relative",
        zIndex: 1,
      }}
    >
      {/* WebGL background */}
      <Background />

      {/* Header bar */}
      <header
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "6px 16px",
          borderBottom: "1px solid var(--border)",
          background: "rgba(11, 18, 32, 0.95)",
          zIndex: 10,
          flexShrink: 0,
        }}
      >
        {/* Title */}
        <div
          className="mono-heading"
          style={{
            fontSize: 14,
            letterSpacing: 4,
            color: "var(--cyan)",
            display: "flex",
            alignItems: "center",
            gap: 10,
          }}
        >
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <polygon
              points="2,8 8,2 14,8 8,14"
              stroke="var(--cyan)"
              strokeWidth="1"
              fill="none"
            />
            <circle cx="8" cy="8" r="2" fill="var(--amber)" />
          </svg>
          MUD-PUPPY STUDIO
        </div>

        {/* Status */}
        <div style={{ display: "flex", alignItems: "center", gap: 16, fontSize: 10 }}>
          {isRunning && (
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <div className="status-dot running" />
              <span
                style={{
                  fontFamily: "'Share Tech Mono', monospace",
                  letterSpacing: "2px",
                  textTransform: "uppercase",
                  color: "var(--lime)",
                }}
              >
                TRAINING
              </span>
            </div>
          )}
          <span
            style={{
              fontFamily: "'Share Tech Mono', monospace",
              letterSpacing: "1px",
              color: "var(--dim)",
            }}
          >
            v0.4.0
          </span>
        </div>
      </header>

      {/* Body: sidebar + pane */}
      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
        {/* Sidebar */}
        <nav
          style={{
            width: 120,
            flexShrink: 0,
            borderRight: "1px solid var(--border)",
            background: "rgba(11, 18, 32, 0.9)",
            display: "flex",
            flexDirection: "column",
            padding: "8px 0",
            zIndex: 5,
          }}
        >
          {PANES.map((p) => {
            const active = p.id === activePane;
            const hasAlert = p.id === "monitor" && isRunning;

            return (
              <button
                key={p.id}
                onClick={() => setActivePane(p.id)}
                style={{
                  width: "100%",
                  padding: "10px 12px",
                  textAlign: "left",
                  background: active ? "rgba(0,229,255,0.08)" : "transparent",
                  border: "none",
                  borderLeft: `2px solid ${active ? "var(--cyan)" : "transparent"}`,
                  color: active ? "var(--cyan)" : "var(--dim)",
                  fontFamily: "'Share Tech Mono', monospace",
                  fontSize: "10px",
                  letterSpacing: "2px",
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                  borderRadius: "0 !important",
                  transition: "all 0.15s",
                }}
              >
                {/* Key hint */}
                <span
                  style={{
                    color: "var(--border)",
                    fontSize: "8px",
                    fontFamily: "'JetBrains Mono', monospace",
                    flexShrink: 0,
                  }}
                >
                  {p.key}
                </span>
                {p.label}
                {hasAlert && (
                  <div
                    className="status-dot running"
                    style={{ width: 5, height: 5, marginLeft: "auto" }}
                  />
                )}
              </button>
            );
          })}

          {/* Spacer */}
          <div style={{ flex: 1 }} />

          {/* Background toggle */}
          <button
            onClick={toggleBackground}
            style={{
              width: "100%",
              padding: "8px 12px",
              textAlign: "left",
              background: "transparent",
              border: "none",
              borderLeft: "2px solid transparent",
              color: "var(--dim)",
              fontFamily: "'Share Tech Mono', monospace",
              fontSize: "8px",
              letterSpacing: "1px",
              cursor: "pointer",
              opacity: 0.6,
            }}
            title="Ctrl+Shift+M"
          >
            [M] BG
          </button>
        </nav>

        {/* Active pane */}
        <main
          style={{
            flex: 1,
            overflow: "hidden",
            background: "rgba(5, 7, 13, 0.85)",
            position: "relative",
            zIndex: 2,
          }}
        >
          <PaneContent pane={activePane} />
        </main>
      </div>
    </div>
  );
}
