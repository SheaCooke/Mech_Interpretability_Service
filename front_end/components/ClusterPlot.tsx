import { useEffect, useRef, useState, useMemo } from "react";
import type { ClusterPoint } from "../api/client";
import type { Label } from "../types";

interface Props {
  points: ClusterPoint[];
  method: string;
}

const PALETTE = [
  "#7c6af7", "#f7836a", "#4ade80", "#facc15",
  "#38bdf8", "#f472b6", "#a78bfa", "#34d399",
  "#fb923c", "#e879f9", "#67e8f9", "#fde047",
  "#86efac", "#fca5a5", "#93c5fd", "#d8b4fe",
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

  // We assign colours by stable insertion order across unique label values.
  // Converting each label to a string key means the map works identically for
  // integer labels (0, 1, 2), float labels (0.5, 1.5), and string labels
  // ("setosa", "versicolor") without any special-casing.
  const colorMap = useMemo(() => {
    const map = new Map<string, string>();
    for (const p of points) {
      const key = String(p.label ?? "__null__");
      if (!map.has(key)) {
        map.set(key, PALETTE[map.size % PALETTE.length]);
      }
    }
    return map;
  }, [points]);

  function colorFor(label: Label | null): string {
    return colorMap.get(String(label ?? "__null__")) ?? "#6b6b80";
  }

  // Sort numerically when all labels are numeric, lexicographically otherwise.
  const legendEntries = useMemo(() => {
    const keys = Array.from(colorMap.keys()).filter(k => k !== "__null__");
    const allNumeric = keys.length > 0 && keys.every(k => !isNaN(Number(k)));
    const sorted = allNumeric
      ? [...keys].sort((a, b) => Number(a) - Number(b))
      : [...keys].sort((a, b) => a.localeCompare(b));
    return sorted.map(key => ({ key, color: colorMap.get(key)! }));
  }, [colorMap]);

  // Computed once per points change and reused in both the draw effect and
  // the mouse move handler to avoid duplicating the min/max/range logic.
  const coordHelpers = useMemo(() => {
    if (points.length === 0) return null;
    const xs = points.map(p => p.x);
    const ys = points.map(p => p.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    return { minX, minY, rangeX, rangeY };
  }, [points]);

  function toCanvas(
    x: number, y: number,
    W: number, H: number,
    helpers: NonNullable<typeof coordHelpers>
  ): [number, number] {
    return [
      PADDING + ((x - helpers.minX) / helpers.rangeX) * (W - PADDING * 2),
      PADDING + ((y - helpers.minY) / helpers.rangeY) * (H - PADDING * 2),
    ];
  }


  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || points.length === 0 || !coordHelpers) return;
    const ctx = canvas.getContext("2d") as CanvasRenderingContext2D;
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;

    ctx.clearRect(0, 0, W, H);

    for (const p of points) {
      const [cx, cy] = toCanvas(p.x, p.y, W, H, coordHelpers);

      ctx.beginPath();
      ctx.arc(cx, cy, POINT_RADIUS, 0, Math.PI * 2);
      ctx.fillStyle = colorFor(p.label);
      ctx.globalAlpha = p.correct ? 0.85 : 0.4;
      ctx.fill();

      if (!p.correct) {
        ctx.beginPath();
        ctx.arc(cx, cy, POINT_RADIUS, 0, Math.PI * 2);
        ctx.strokeStyle = "#f87171";
        ctx.lineWidth = 1.5;
        ctx.globalAlpha = 0.6;
        ctx.stroke();
      }

      ctx.globalAlpha = 1;
    }
  }, [points, coordHelpers, colorMap]);


  function handleMouseMove(e: React.MouseEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas || points.length === 0 || !coordHelpers) return;

    const rect   = canvas.getBoundingClientRect();
    //Scale mouse position from CSS pixels to canvas pixels
    const scaleX = canvas.width  / rect.width;
    const scaleY = canvas.height / rect.height;
    const mx = (e.clientX - rect.left) * scaleX;
    const my = (e.clientY - rect.top)  * scaleY;

    let closest: ClusterPoint | null = null;
    let closestDist = HOVER_RADIUS;

    for (const p of points) {
      const [cx, cy] = toCanvas(p.x, p.y, canvas.width, canvas.height, coordHelpers);
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
            <div>label: {String(tooltip.point.label ?? "—")}</div>
            <div>predicted: {String(tooltip.point.predicted)}</div>
            <div style={{
              color: tooltip.point.correct ? "var(--success)" : "var(--error)",
            }}>
              {tooltip.point.correct ? "✓ correct" : "✗ incorrect"}
            </div>
          </div>
        )}
      </div>

      <div style={{ display: "flex", flexWrap: "wrap", gap: "10px" }}>
        {legendEntries.map(({ key, color }) => (
          <div
            key={key}
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
              background: color,
              display: "inline-block", flexShrink: 0,
            }} />
            {key}
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