/**
 * Panel — clipped-corner vector frame with amber corner brackets.
 *
 * Structure follows the 2026-04-22 design handoff:
 *   - CSS clip-path creates the 8px 45-degree corner cuts
 *   - Absolute SVG overlay draws the 1px border + amber accent brackets
 *   - Optional floating label in the top-left of the border
 *   - Inline CSS classes (.panel / .panel-label / .panel-body) live in
 *     styles/index.css so the markup stays thin.
 */
import React from "react";

interface PanelProps {
  children: React.ReactNode;
  /** Floating label rendered in the top border (uppercase micro-heading). */
  label?: string;
  /** Active state — adds cyan outer glow via drop-shadow filter. */
  active?: boolean;
  /** Alert state — adds magenta outer glow. */
  alert?: boolean;
  className?: string;
  style?: React.CSSProperties;
  /** Style override for the inner padded body. */
  bodyStyle?: React.CSSProperties;
  id?: string;
}

export function Panel({
  children,
  label,
  active = false,
  alert = false,
  className = "",
  style,
  bodyStyle,
  id,
}: PanelProps) {
  const borderColor = alert ? "var(--magenta)" : "var(--cyan)";
  const cls = [
    "panel",
    active ? "active" : "",
    alert ? "alert" : "",
    className,
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div id={id} className={cls} style={style}>
      <div className="frame">
        <svg
          preserveAspectRatio="none"
          viewBox="0 0 100 100"
          aria-hidden="true"
        >
          {/* Main outline — 1px stroke running along the clipped edges */}
          <polygon
            points="8,0.5 92,0.5 99.5,8 99.5,92 92,99.5 8,99.5 0.5,92 0.5,8"
            fill="none"
            stroke={borderColor}
            strokeWidth="0.5"
            vectorEffect="non-scaling-stroke"
            opacity="0.7"
          />
          {/* Top-left amber bracket */}
          <polyline
            points="14,0.5 0.5,0.5 0.5,14"
            fill="none"
            stroke="var(--amber)"
            strokeWidth="1.5"
            strokeLinecap="square"
            vectorEffect="non-scaling-stroke"
          />
          {/* Bottom-right amber bracket */}
          <polyline
            points="86,99.5 99.5,99.5 99.5,86"
            fill="none"
            stroke="var(--amber)"
            strokeWidth="1.5"
            strokeLinecap="square"
            vectorEffect="non-scaling-stroke"
          />
        </svg>
      </div>
      {label && <div className="panel-label">{label}</div>}
      <div className="panel-body" style={bodyStyle}>
        {children}
      </div>
    </div>
  );
}
