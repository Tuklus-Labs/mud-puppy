/**
 * Background — static SVG grid with radial vignette.
 *
 * Per the 2026-04-22 design handoff this replaced the WebGL phase-field
 * shader. It is static by design: zero animation work, zero GPU, zero noise
 * on prefers-reduced-motion.  The `enabled` store flag is retained so the
 * existing `Ctrl+Shift+M` toggle still turns the grid off entirely.
 */
import React from "react";
import { useStore } from "../lib/store";

export function Background() {
  const enabled = useStore((s) => s.backgroundEnabled);
  if (!enabled) return null;

  return (
    <svg className="bg-field" aria-hidden="true" width="100%" height="100%">
      <defs>
        <pattern id="grid" width="48" height="48" patternUnits="userSpaceOnUse">
          <path
            d="M 48 0 L 0 0 0 48"
            fill="none"
            stroke="#17263b"
            strokeWidth="0.5"
          />
        </pattern>
        <radialGradient id="vignette" cx="50%" cy="50%" r="70%">
          <stop offset="0%" stopColor="rgba(0,229,255,0.04)" />
          <stop offset="60%" stopColor="rgba(0,0,0,0)" />
          <stop offset="100%" stopColor="rgba(0,0,0,0.6)" />
        </radialGradient>
      </defs>
      <rect width="100%" height="100%" fill="url(#grid)" />
      <rect width="100%" height="100%" fill="url(#vignette)" />
    </svg>
  );
}
