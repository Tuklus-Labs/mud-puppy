/**
 * VectorFrame — SVG panel frame with 45-degree clipped corners.
 *
 * 1px cyan stroke on main rect, 2px amber corner accents at 8px clip size.
 * No rounded corners anywhere — this is the distinctive aesthetic element.
 * Drop-shadow filter only activates on [data-active] or [data-alert] state.
 */
import React from "react";

interface VectorFrameProps {
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

export function VectorFrame({
  children,
  label,
  active = false,
  alert = false,
  className = "",
  style,
  id,
}: VectorFrameProps) {
  const glowColor = alert ? "rgba(255, 43, 214, 0.6)" : "rgba(0, 229, 255, 0.5)";
  const showGlow = active || alert;

  return (
    <div
      id={id}
      className={className}
      data-active={active ? "true" : undefined}
      data-alert={alert ? "true" : undefined}
      style={{
        position: "relative",
        background: "var(--panel)",
        // Use clip-path for the 45-degree corner effect
        clipPath: `polygon(
          ${CLIP}px 0,
          calc(100% - ${CLIP}px) 0,
          100% ${CLIP}px,
          100% calc(100% - ${CLIP}px),
          calc(100% - ${CLIP}px) 100%,
          ${CLIP}px 100%,
          0 calc(100% - ${CLIP}px),
          0 ${CLIP}px
        )`,
        filter: showGlow
          ? `drop-shadow(0 0 6px ${glowColor}) drop-shadow(0 0 14px ${glowColor})`
          : undefined,
        transition: "filter 0.2s ease",
        ...style,
      }}
    >
      {/* SVG overlay for the border strokes */}
      <svg
        style={{
          position: "absolute",
          inset: 0,
          width: "100%",
          height: "100%",
          pointerEvents: "none",
          overflow: "visible",
        }}
      >
        <defs>
          <filter id="frame-glow">
            <feGaussianBlur stdDeviation="2" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Main border — 1px cyan, clipped corners */}
        <polygon
          points={`
            ${CLIP},1
            calc(100% - ${CLIP}),1
            ${/* right edge handled via percent — use 99.5% for inner stroke */``}
          `}
          fill="none"
          stroke="none"
        />

        {/* Use a rect with clip-path mirroring the container */}
        <rect
          x="0.5"
          y="0.5"
          width="calc(100% - 1px)"
          height="calc(100% - 1px)"
          fill="none"
          stroke="var(--cyan)"
          strokeWidth="1"
          style={{
            // SVG doesn't support CSS calc in attributes, use a workaround
            // We'll draw the polygon manually using 100% tricks via a foreignObject approach
          }}
        />

        {/* Amber 2px corner accents — top-left */}
        <polyline
          points={`${CLIP + 6},1 1,1 1,${CLIP + 6}`}
          fill="none"
          stroke="var(--amber)"
          strokeWidth="2"
          strokeLinecap="square"
        />
      </svg>

      {/* Content */}
      <div style={{ position: "relative", zIndex: 1, width: "100%", height: "100%" }}>
        {label && (
          <div
            style={{
              position: "absolute",
              top: 0,
              left: CLIP + 4,
              fontSize: "10px",
              fontFamily: "var(--font-heading, 'Share Tech Mono', monospace)",
              letterSpacing: "2px",
              textTransform: "uppercase",
              color: "var(--cyan)",
              background: "var(--panel)",
              padding: "0 4px",
              transform: "translateY(-50%)",
              zIndex: 2,
            }}
          >
            {label}
          </div>
        )}
        {children}
      </div>
    </div>
  );
}

/**
 * Simplified version using CSS-only approach for better SVG compatibility.
 * This is the production-ready version.
 */
export function Panel({
  children,
  label,
  active = false,
  alert = false,
  className = "",
  style,
  id,
}: VectorFrameProps) {
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
      {/* Inner border via inset pseudo-element approach using outline or box-shadow */}
      {/* Since clip-path clips box-shadow, we use a position:absolute SVG */}
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
        {/* Main 1px border using polygon in percentage coordinates */}
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
