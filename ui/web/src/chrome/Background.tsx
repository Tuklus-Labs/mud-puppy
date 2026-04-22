/**
 * Background — WebGL phase-field shader (twgl.js).
 *
 * Fragment shader uses fbm noise * slow time for a dark phase-grid aesthetic
 * with cyan highlights. 30fps cap via rAF timestamp diff.
 *
 * Disableable via:
 *   - ?chrome=minimal URL param
 *   - Ctrl+Shift+M keyboard shortcut (persists to localStorage)
 */
import React, { useEffect, useRef } from "react";
import * as twgl from "twgl.js";
import { useStore } from "../lib/store";

// ---------------------------------------------------------------------------
// GLSL shaders
// ---------------------------------------------------------------------------

const VS = /* glsl */ `
  attribute vec4 position;
  void main() {
    gl_Position = position;
  }
`;

const FS = /* glsl */ `
  precision mediump float;
  uniform float uTime;
  uniform vec2  uResolution;

  // ---- Noise / fbm helpers ----
  float hash(vec2 p) {
    p = fract(p * vec2(234.34, 435.345));
    p += dot(p, p + 34.23);
    return fract(p.x * p.y);
  }

  float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(
      mix(hash(i), hash(i + vec2(1,0)), u.x),
      mix(hash(i + vec2(0,1)), hash(i + vec2(1,1)), u.x),
      u.y
    );
  }

  float fbm(vec2 p) {
    float v = 0.0;
    float a = 0.5;
    for (int i = 0; i < 5; i++) {
      v += a * noise(p);
      p  = p * 2.0 + vec2(1.7, 9.2);
      a *= 0.5;
    }
    return v;
  }

  void main() {
    vec2 uv = gl_FragCoord.xy / uResolution;
    uv.y = 1.0 - uv.y; // flip Y

    // Slow time - phase field feel
    float t = uTime * 0.04;

    // Phase field: fbm-based noise with slow drift
    vec2 p = uv * 4.0;
    float n = fbm(p + t);
    float n2 = fbm(p * 1.5 + t * 0.7 + 3.1);

    // Grid lines at phase boundaries
    float phase = sin(n * 6.28 * 2.0 + t * 0.5);
    float grid = abs(phase);
    grid = smoothstep(0.85, 1.0, grid) * 0.3;

    // Faint base field color — near-void dark blue
    vec3 baseColor = vec3(0.02, 0.04, 0.08);

    // Cyan highlight on grid lines
    vec3 cyanColor = vec3(0.0, 0.9, 1.0);
    vec3 gridColor = mix(baseColor, cyanColor, grid * 0.6);

    // Subtle energy variation
    float energy = fbm(uv * 8.0 + t * 0.3) * 0.15;
    vec3 finalColor = gridColor + vec3(energy * 0.02, energy * 0.04, energy * 0.06);

    // Vignette
    float vignette = 1.0 - dot(uv - 0.5, uv - 0.5) * 1.2;
    finalColor *= max(vignette, 0.1);

    // Keep very dark — this is a subtle bg, not a light show
    finalColor = mix(baseColor, finalColor, 0.7);

    gl_FragColor = vec4(finalColor, 1.0);
  }
`;

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function Background() {
  const enabled = useStore((s) => s.backgroundEnabled);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const glRef = useRef<WebGLRenderingContext | null>(null);
  const rafRef = useRef<number>(0);
  const lastFrameRef = useRef<number>(0);
  const programInfoRef = useRef<twgl.ProgramInfo | null>(null);
  const bufferInfoRef = useRef<twgl.BufferInfo | null>(null);

  useEffect(() => {
    if (!enabled || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const gl = canvas.getContext("webgl");
    if (!gl) return;
    glRef.current = gl;

    const programInfo = twgl.createProgramInfo(gl, [VS, FS]);
    programInfoRef.current = programInfo;

    const arrays = {
      position: {
        numComponents: 2,
        data: new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
      },
    };
    bufferInfoRef.current = twgl.createBufferInfoFromArrays(gl, arrays);

    let startTime = performance.now();

    function render(now: number) {
      if (!gl || !programInfoRef.current || !bufferInfoRef.current) return;

      // 30fps cap
      const elapsed = now - lastFrameRef.current;
      if (elapsed < 33.3) {
        rafRef.current = requestAnimationFrame(render);
        return;
      }
      lastFrameRef.current = now;

      // Resize if needed
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
        gl.viewport(0, 0, w, h);
      }

      const t = (now - startTime) / 1000;

      gl.useProgram(programInfoRef.current.program);
      twgl.setBuffersAndAttributes(gl, programInfoRef.current, bufferInfoRef.current);
      twgl.setUniforms(programInfoRef.current, {
        uTime: t,
        uResolution: [canvas.width, canvas.height],
      });
      twgl.drawBufferInfo(gl, bufferInfoRef.current);

      rafRef.current = requestAnimationFrame(render);
    }

    rafRef.current = requestAnimationFrame(render);

    return () => {
      cancelAnimationFrame(rafRef.current);
    };
  }, [enabled]);

  if (!enabled) return null;

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: "fixed",
        inset: 0,
        width: "100%",
        height: "100%",
        zIndex: 0,
        pointerEvents: "none",
        opacity: 0.7,
      }}
    />
  );
}
