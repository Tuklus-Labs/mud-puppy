/**
 * Shell — header, 180px nav rail, main pane area, 28px footer status bar.
 * Layout per the 2026-04-22 design handoff.
 */
import React, { useEffect, useCallback } from "react";
import { useStore } from "./lib/store";
import { ipc } from "./lib/ipc";
import { Background } from "./chrome/Background";
import { BrandMark, NavIcon } from "./chrome/BrandMark";
import { Launch } from "./panes/Launch";
import { Monitor } from "./panes/Monitor";
import { Runs } from "./panes/Runs";
import { Library } from "./panes/Library";
import { Logs } from "./panes/Logs";

type Pane = "launch" | "monitor" | "runs" | "library" | "logs";

const NAV: { id: Pane; label: string; key: string }[] = [
  { id: "launch",  label: "Launch",  key: "1" },
  { id: "monitor", label: "Monitor", key: "2" },
  { id: "runs",    label: "Runs",    key: "3" },
  { id: "library", label: "Library", key: "4" },
  { id: "logs",    label: "Logs",    key: "5" },
];

export function App() {
  const activePane = useStore((s) => s.activePane);
  const setActivePane = useStore((s) => s.setActivePane);
  const toggleBackground = useStore((s) => s.toggleBackground);
  const runs = useStore((s) => s.runs);
  const activeRunId = useStore((s) => s.activeRunId);

  const activeRun = activeRunId ? runs.find((r) => r.run_id === activeRunId) : null;
  const isRunning = activeRun?.status === "running";

  // Subscribe to all IPC events at the App level so they flow to the store
  // regardless of which pane is active. Monitor and Logs both read from the
  // store; subscribing here means events are never missed during pane switches.
  useEffect(() => {
    const store = useStore.getState();
    const unsubs = [
      ipc.onMetrics((m) => store.appendMetrics(m)),
      ipc.onGpu((g) => store.appendGpu(g)),
      ipc.onStreamStats((s) => store.setStreamStats(s)),
      ipc.onMemoryStats((s) => store.setMemoryStats(s)),
      ipc.onLogLine((l) => store.appendLog(l)),
      ipc.onRunComplete((e) => {
        // Mark the matching run as complete/failed in the run list.
        const current = useStore.getState().runs;
        const idx = current.findIndex((r) => r.run_id === e.run_id);
        if (idx >= 0) {
          const updated = [...current];
          updated[idx] = {
            ...updated[idx],
            status: e.exit_code === 0 ? "complete" : "failed",
            end_time: Date.now(),
          };
          useStore.setState({ runs: updated });
        }
      }),
    ];
    return () => unsubs.forEach((u) => u());
  }, []);

  const handleKey = useCallback(
    (e: KeyboardEvent) => {
      if (e.ctrlKey && e.shiftKey && e.key === "M") {
        e.preventDefault();
        toggleBackground();
        return;
      }
      if (!e.ctrlKey && !e.altKey && !e.metaKey) {
        const hit = NAV.find((n) => n.key === e.key);
        if (
          hit &&
          !(e.target instanceof HTMLInputElement) &&
          !(e.target instanceof HTMLTextAreaElement)
        ) {
          e.preventDefault();
          setActivePane(hit.id);
        }
      }
    },
    [setActivePane, toggleBackground],
  );

  useEffect(() => {
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [handleKey]);

  let body: React.ReactNode = null;
  if (activePane === "launch") body = <Launch />;
  else if (activePane === "monitor") body = <Monitor />;
  else if (activePane === "runs") body = <Runs />;
  else if (activePane === "library") body = <Library />;
  else body = <Logs />;

  return (
    <div className="shell scanlines">
      <Background />

      <header className="header">
        <div className="brand">
          <BrandMark size={22} />
          <div className="brand-name">MUD · PUPPY</div>
          <div className="brand-sep" />
          <div className="brand-sub">Fine-Tune Studio</div>
        </div>
        <div className="header-right">
          {isRunning && activeRun && (
            <div className="header-stat">
              <span className="dot running" />
              <span className="label-inline" style={{ color: "var(--lime)" }}>
                Training
              </span>
              {activeRun.steps_total && activeRun.steps_done != null && (
                <span className="val">
                  step {activeRun.steps_done.toLocaleString()} /{" "}
                  {activeRun.steps_total.toLocaleString()}
                </span>
              )}
            </div>
          )}
          <div className="ver">v0.4.0</div>
        </div>
      </header>

      <nav className="nav">
        <div className="nav-section">Workspace</div>
        {NAV.map((n) => {
          const active = activePane === n.id;
          return (
            <button
              key={n.id}
              className={`nav-item ${active ? "active" : ""}`}
              onClick={() => setActivePane(n.id)}
            >
              <NavIcon id={n.id} active={active} />
              {n.label}
              {n.id === "monitor" && isRunning ? (
                <span className="nav-alert" />
              ) : (
                <span className="nav-key">{n.key}</span>
              )}
            </button>
          );
        })}
        <div style={{ flex: 1 }} />
        <div className="nav-section" style={{ marginTop: 0 }}>
          System
        </div>
        <button
          className="nav-item"
          onClick={toggleBackground}
          title="Toggle background (Ctrl+Shift+M)"
          aria-label="Toggle background animation"
        >
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <circle cx="7" cy="7" r="2.2" stroke="currentColor" strokeWidth="0.8" />
            <circle cx="7" cy="7" r="5.5" stroke="currentColor" strokeWidth="0.8" />
            <line x1="7" y1="0.5" x2="7" y2="2.5" stroke="currentColor" strokeWidth="0.8" />
            <line x1="7" y1="11.5" x2="7" y2="13.5" stroke="currentColor" strokeWidth="0.8" />
            <line x1="0.5" y1="7" x2="2.5" y2="7" stroke="currentColor" strokeWidth="0.8" />
            <line x1="11.5" y1="7" x2="13.5" y2="7" stroke="currentColor" strokeWidth="0.8" />
          </svg>
          Display
        </button>
      </nav>

      <main className="main">{body}</main>

      <footer className="foot">
        <div className="foot-group">
          <span>
            <b>mud-puppy</b>
          </span>
          <span className="foot-sep">·</span>
          <span>
            daemon{" "}
            <span style={{ color: "var(--lime)" }}>
              {isRunning ? "active" : "idle"}
            </span>
          </span>
        </div>
        <div className="foot-group">
          <span>ROCm 7.1</span>
          <span className="foot-sep">·</span>
          <span>torch 2.10</span>
          <span className="foot-sep">·</span>
          <span>
            <span className="kbd">1</span>–<span className="kbd">5</span> navigate
          </span>
          <span className="foot-sep">·</span>
          <span>
            <span className="kbd">Ctrl+Shift+M</span> bg
          </span>
        </div>
      </footer>
    </div>
  );
}
