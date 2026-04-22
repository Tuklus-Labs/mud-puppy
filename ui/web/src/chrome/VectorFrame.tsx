/**
 * Panel — SVG-edged panel frame with 45-degree clipped corners.
 *
 * This is the sole production component exported from this file. The
 * earlier VectorFrame export was dead code (broken SVG polygons) and has
 * been removed.
 *
 * Aesthetic:
 * - clip-path polygon for the 8px corner cut (no rounded corners anywhere)
 * - 1px cyan stroke via inline SVG viewBox polyline
 * - 2px amber corner accents (top-left + bottom-right)
 * - drop-shadow filter only active on [data-active] / [data-alert]
 */
import React from "react";

interface PanelProps {
  children: React.ReactNode;
  /** Optional label displayed in the top-left corner accent */
  label?: string;
  /** Active state — enables outer glow */
  active?: boolean;
  /** Alert state — enables magenta glow */
  alert?: boolean;
  className?: string;
  style?: React.CSSProperties;
  id?: string;
}

// Corner clip size in pixels
const CLIP = 8;

export function Panel({
  children,
  label,
  active = false,
  alert = false,
  className = "",
  style,
  id,
}: PanelProps) {
  const cornerSize = CLIP;
  const glowColor = alert ? "rgba(255, 43, 214, 0.55)" : "rgba(0, 229, 255, 0.45)";
  const showGlow = active || alert;
  const borderColor = alert ? "var(--magenta)" : "var(--cyan)";

  return (
    <div
      id={id}
      className={className}
      style={{
        position: "relative",
        background: "var(--panel)",
        clipPath: `polygon(
          ${cornerSize}px 0%,
          calc(100% - ${cornerSize}px) 0%,
          100% ${cornerSize}px,
          100% calc(100% - ${cornerSize}px),
          calc(100% - ${cornerSize}px) 100%,
          ${cornerSize}px 100%,
          0% calc(100% - ${cornerSize}px),
          0% ${cornerSize}px
        )`,
        filter: showGlow
          ? `drop-shadow(0 0 5px ${glowColor}) drop-shadow(0 0 12px ${glowColor})`
          : undefined,
        transition: "filter 0.25s ease",
        ...style,
      }}
    >
      {/* Border + corner accents via absolute SVG (clip-path eats box-shadow) */}
      <svg
        aria-hidden="true"
        style={{
          position: "absolute",
          inset: 0,
          width: "100%",
          height: "100%",
          pointerEvents: "none",
          zIndex: 10,
          overflow: "visible",
        }}
        preserveAspectRatio="none"
        viewBox="0 0 100 100"
      >
        {/* Main 1px border — polygon in percentage coordinates */}
        <polygon
          points={`
            ${cornerSize},0.5
            ${100 - cornerSize},0.5
            99.5,${cornerSize}
            99.5,${100 - cornerSize}
            ${100 - cornerSize},99.5
            ${cornerSize},99.5
            0.5,${100 - cornerSize}
            0.5,${cornerSize}
          `}
          fill="none"
          stroke={borderColor}
          strokeWidth="0.5"
          vectorEffect="non-scaling-stroke"
        />

        {/* Top-left corner accent — 2px amber */}
        <polyline
          points={`${cornerSize + 4},0.5 0.5,0.5 0.5,${cornerSize + 4}`}
          fill="none"
          stroke="var(--amber)"
          strokeWidth="1.5"
          strokeLinecap="square"
          vectorEffect="non-scaling-stroke"
        />

        {/* Bottom-right corner accent — 2px amber */}
        <polyline
          points={`${100 - cornerSize - 4},99.5 99.5,99.5 99.5,${100 - cornerSize - 4}`}
          fill="none"
          stroke="var(--amber)"
          strokeWidth="1.5"
          strokeLinecap="square"
          vectorEffect="non-scaling-stroke"
        />
      </svg>

      {/* Label badge */}
      {label && (
        <div
          style={{
            position: "absolute",
            top: "1px",
            left: `${cornerSize + 8}px`,
            fontSize: "9px",
            fontFamily: "'Share Tech Mono', monospace",
            letterSpacing: "2px",
            textTransform: "uppercase",
            color: "var(--cyan)",
            background: "var(--panel)",
            padding: "1px 6px",
            zIndex: 11,
          }}
        >
          {label}
        </div>
      )}

      {/* Content area */}
      <div style={{ position: "relative", zIndex: 1, width: "100%", height: "100%" }}>
        {children}
      </div>
    </div>
  );
}
