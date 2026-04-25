import { useEffect, useRef, useState } from "react";
import type { ClusterPoint } from "../api/client";

interface Props {
  points: ClusterPoint[];
  method: string;
}

const PALETTE = [
  "#7c6af7", "#f7836a", "#4ade80", "#facc15",
  "#38bdf8", "#f472b6", "#a78bfa", "#34d399",
  "#fb923c", "#e879f9",
];

const POINT_RADIUS = 4;
const HOVER_RADIUS = 7;
const PADDING      = 48;

export default function ClusterPlot({ points, method }: Props) {
  const canvasRef  = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLDivElement>(null);
  const [tooltip, setTooltip] = useState<{
    x: number; y: number; point: ClusterPoint;
  } | null>(null);

  // Collect unique labels for the legend
  const labels = Array.from(
    new Set(points.map(p => p.label).filter(l => l !== null))
  ).sort((a, b) => (a as number) - (b as number)) as number[];

  function colorFor(label: number | null): string {
    if (label === null) return "#6b6b80";
    return PALETTE[label % PALETTE.length];
  }

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || points.length === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;

    // Map data coords → canvas coords
    const xs = points.map(p => p.x);
    const ys = points.map(p => p.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;

    function toCanvas(x: number, y: number): [number, number] {
      return [
        PADDING + ((x - minX) / rangeX) * (W - PADDING * 2),
        PADDING + ((y - minY) / rangeY) * (H - PADDING * 2),
      ];
    }

    // Clear
    ctx.clearRect(0, 0, W, H);

    // Draw points
    for (const p of points) {
      const [cx, cy] = toCanvas(p.x, p.y);
      ctx.beginPath();
      ctx.arc(cx, cy, POINT_RADIUS, 0, Math.PI * 2);
      ctx.fillStyle = colorFor(p.label);
      ctx.globalAlpha = p.correct ? 0.85 : 0.4;
      ctx.fill();

      // Ring around incorrect predictions
      if (!p.correct) {
        ctx.strokeStyle = "#f87171";
        ctx.lineWidth = 1.5;
        ctx.globalAlpha = 0.6;
        ctx.stroke();
      }

      ctx.globalAlpha = 1;
    }
  }, [points]);

  function handleMouseMove(e: React.MouseEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas || points.length === 0) return;

    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const W = canvas.width;
    const H = canvas.height;

    const xs = points.map(p => p.x);
    const ys = points.map(p => p.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;

    function toCanvas(x: number, y: number): [number, number] {
      return [
        PADDING + ((x - minX) / rangeX) * (W - PADDING * 2),
        PADDING + ((y - minY) / rangeY) * (H - PADDING * 2),
      ];
    }

    let closest: ClusterPoint | null = null;
    let closestDist = HOVER_RADIUS;

    for (const p of points) {
      const [cx, cy] = toCanvas(p.x, p.y);
      const d = Math.sqrt((mx - cx) ** 2 + (my - cy) ** 2);
      if (d < closestDist) { closestDist = d; closest = p; }
    }

    if (closest) {
      setTooltip({ x: e.clientX, y: e.clientY, point: closest });
    } else {
      setTooltip(null);
    }
  }

  return (
    <div className="panel">
      <h3 className="panel-title">
        <span style={{ fontSize: 14 }}>⬡</span> Activation Cluster Plot ({method})
      </h3>

      <div style={{ position: "relative" }}>
        <canvas
          ref={canvasRef}
          width={720}
          height={480}
          style={{
            width: "100%",
            borderRadius: "var(--radius)",
            background: "var(--bg)",
            border: "1px solid var(--border)",
            cursor: "crosshair",
          }}
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setTooltip(null)}
        />

        {/* Tooltip */}
        {tooltip && (
          <div
            ref={overlayRef}
            style={{
              position: "fixed",
              left: tooltip.x + 12,
              top:  tooltip.y - 8,
              background: "var(--surface)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius)",
              padding: "6px 10px",
              fontSize: 11,
              fontFamily: "var(--mono)",
              color: "var(--text)",
              pointerEvents: "none",
              zIndex: 100,
              whiteSpace: "nowrap",
            }}
          >
            <div style={{ color: "var(--accent)", marginBottom: 2 }}>
              {tooltip.point.id}
            </div>
            <div>label: {tooltip.point.label ?? "—"}</div>
            <div>predicted: {tooltip.point.predicted}</div>
            <div style={{ color: tooltip.point.correct ? "var(--success)" : "var(--error)" }}>
              {tooltip.point.correct ? "✓ correct" : "✗ incorrect"}
            </div>
          </div>
        )}
      </div>

      {/* Legend */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: "10px" }}>
        {labels.map(label => (
          <div
            key={label}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 6,
              fontSize: 11,
              fontFamily: "var(--mono)",
              color: "var(--muted)",
            }}
          >
            <span style={{
              width: 10, height: 10, borderRadius: "50%",
              background: colorFor(label),
              display: "inline-block", flexShrink: 0,
            }} />
            Label {label}
          </div>
        ))}
        <div style={{
          display: "flex", alignItems: "center", gap: 6,
          fontSize: 11, fontFamily: "var(--mono)", color: "var(--muted)",
        }}>
          <span style={{
            width: 10, height: 10, borderRadius: "50%",
            background: "transparent",
            border: "1.5px solid #f87171",
            display: "inline-block", flexShrink: 0,
          }} />
          Incorrect prediction
        </div>
      </div>
    </div>
  );
}