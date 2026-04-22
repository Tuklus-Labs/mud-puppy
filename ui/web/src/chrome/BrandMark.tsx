/**
 * Brand mark + nav glyph icons — small SVG vector graphics used in the shell.
 */
import React from "react";

export function BrandMark({ size = 22 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      className="brand-mark"
      aria-hidden="true"
    >
      <polygon
        points="12,2 22,7 22,17 12,22 2,17 2,7"
        fill="none"
        stroke="var(--cyan)"
        strokeWidth="1.2"
      />
      <polygon
        points="12,6 18,9 18,15 12,18 6,15 6,9"
        fill="none"
        stroke="var(--cyan)"
        strokeWidth="0.8"
        opacity="0.6"
      />
      <circle cx="12" cy="12" r="2" fill="var(--amber)" />
      <line
        x1="12"
        y1="2"
        x2="12"
        y2="6"
        stroke="var(--cyan)"
        strokeWidth="0.6"
        opacity="0.5"
      />
      <line
        x1="12"
        y1="18"
        x2="12"
        y2="22"
        stroke="var(--cyan)"
        strokeWidth="0.6"
        opacity="0.5"
      />
    </svg>
  );
}

type IconKind = "launch" | "monitor" | "runs" | "library" | "logs";

export function NavIcon({ id, active = false }: { id: IconKind; active?: boolean }) {
  const c = "currentColor";
  const sz = 14;
  if (id === "launch")
    return (
      <svg width={sz} height={sz} viewBox="0 0 14 14" fill="none">
        <polygon
          points="2,7 12,1 12,13"
          stroke={c}
          strokeWidth="0.9"
          fill={active ? c : "none"}
          fillOpacity="0.15"
        />
      </svg>
    );
  if (id === "monitor")
    return (
      <svg width={sz} height={sz} viewBox="0 0 14 14" fill="none">
        <polyline
          points="1,9 4,6 7,8 10,3 13,5"
          stroke={c}
          strokeWidth="1"
          fill="none"
        />
        <line
          x1="0.5"
          y1="12.5"
          x2="13.5"
          y2="12.5"
          stroke={c}
          strokeWidth="0.6"
          opacity="0.5"
        />
      </svg>
    );
  if (id === "runs")
    return (
      <svg width={sz} height={sz} viewBox="0 0 14 14" fill="none">
        <rect x="1" y="2.5" width="12" height="2" stroke={c} strokeWidth="0.8" />
        <rect x="1" y="6" width="12" height="2" stroke={c} strokeWidth="0.8" />
        <rect x="1" y="9.5" width="12" height="2" stroke={c} strokeWidth="0.8" />
      </svg>
    );
  if (id === "library")
    return (
      <svg width={sz} height={sz} viewBox="0 0 14 14" fill="none">
        <rect x="1" y="3" width="3.5" height="9" stroke={c} strokeWidth="0.8" />
        <rect x="5.5" y="2" width="3.5" height="10" stroke={c} strokeWidth="0.8" />
        <rect x="10" y="4" width="3" height="8" stroke={c} strokeWidth="0.8" />
      </svg>
    );
  return (
    <svg width={sz} height={sz} viewBox="0 0 14 14" fill="none">
      <line x1="2" y1="3" x2="12" y2="3" stroke={c} strokeWidth="0.8" />
      <line x1="2" y1="6" x2="9" y2="6" stroke={c} strokeWidth="0.8" />
      <line x1="2" y1="9" x2="11" y2="9" stroke={c} strokeWidth="0.8" />
      <line x1="2" y1="12" x2="7" y2="12" stroke={c} strokeWidth="0.8" />
    </svg>
  );
}
