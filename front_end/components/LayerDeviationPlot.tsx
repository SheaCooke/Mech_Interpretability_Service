import { useEffect, useRef, useState } from "react";
import type { IncorrectRecord, LayerDeviationData } from "../api/client";
import { Label } from "../types";

interface Props {
  sessionId: string;
  incorrectRecords: IncorrectRecord[];
  onRecordSelect: (recordId: string) => void;
  deviationData: LayerDeviationData | null;
  loading: boolean;
}

// ── Chart constants ────────────────────────────────────────────────────────

const PAD    = { top: 24, right: 24, bottom: 64, left: 52 };
const COLOR_TRUE      = "#7c6af7"; // accent — deviation from true-label prototype
const COLOR_PREDICTED = "#f7836a"; // accent2 — deviation from predicted-label prototype

// ── Helpers ────────────────────────────────────────────────────────────────

function drawChart(
  canvas: HTMLCanvasElement,
  data: LayerDeviationData,
) {
  const ctx = canvas.getContext("2d") as CanvasRenderingContext2D;
  if (!ctx) return;

  const W = canvas.width;
  const H = canvas.height;
  const chartW = W - PAD.left - PAD.right;
  const chartH = H - PAD.top  - PAD.bottom;

  ctx.clearRect(0, 0, W, H);

  const layers = data.layer_names;
  const n      = layers.length;
  if (n === 0) return;

  // Filter nulls for scale calculation
  const allVals = [
    ...data.true_label_deviations.filter((v): v is number => v !== null),
    ...data.predicted_deviations.filter((v): v is number => v !== null),
  ];
  if (allVals.length === 0) return;

  const maxVal = Math.max(...allVals, 0.01);
  const minVal = 0;

  function xOf(i: number)   { return PAD.left + (i / (n - 1)) * chartW; }
  function yOf(v: number)   { return PAD.top  + (1 - (v - minVal) / (maxVal - minVal)) * chartH; }

  // ── Grid lines ────────────────────────────────────────────────────────────
  ctx.strokeStyle = "rgba(30,30,46,0.8)";
  ctx.lineWidth   = 1;
  const gridSteps = 4;
  for (let g = 0; g <= gridSteps; g++) {
    const val = minVal + (g / gridSteps) * maxVal;
    const y   = yOf(val);
    ctx.beginPath();
    ctx.moveTo(PAD.left, y);
    ctx.lineTo(W - PAD.right, y);
    ctx.stroke();

    // Y axis labels
    ctx.fillStyle    = "rgba(107,107,128,0.9)";
    ctx.font         = "10px 'JetBrains Mono', monospace";
    ctx.textAlign    = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(val.toFixed(2), PAD.left - 6, y);
  }

  // ── Draw a line series ────────────────────────────────────────────────────
  function drawLine(values: (number | null)[], color: string) {
    ctx.strokeStyle = color;
    ctx.lineWidth   = 2;
    ctx.lineJoin    = "round";
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < n; i++) {
      const v = values[i];
      if (v === null) { started = false; continue; }
      const x = xOf(i);
      const y = yOf(v);
      if (!started) { ctx.moveTo(x, y); started = true; }
      else           { ctx.lineTo(x, y); }
    }
    ctx.stroke();

    // Dots
    for (let i = 0; i < n; i++) {
      const v = values[i];
      if (v === null) continue;
      ctx.beginPath();
      ctx.arc(xOf(i), yOf(v), 3.5, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    }
  }

  drawLine(data.predicted_deviations,    COLOR_PREDICTED);
  drawLine(data.true_label_deviations,   COLOR_TRUE);

  // ── X axis layer labels ───────────────────────────────────────────────────
  ctx.fillStyle    = "rgba(107,107,128,0.9)";
  ctx.font         = "9px 'JetBrains Mono', monospace";
  ctx.textAlign    = "center";
  ctx.textBaseline = "top";

  // Only label every Nth layer to avoid crowding
  const labelEvery = Math.ceil(n / 10);
  for (let i = 0; i < n; i++) {
    if (i % labelEvery !== 0 && i !== n - 1) continue;
    const x     = xOf(i);
    const label = layers[i].length > 10 ? layers[i].slice(0, 9) + "…" : layers[i];
    ctx.save();
    ctx.translate(x, PAD.top + chartH + 6);
    ctx.rotate(-Math.PI / 4);
    ctx.fillText(label, 0, 0);
    ctx.restore();
  }

  // ── Axes ──────────────────────────────────────────────────────────────────
  ctx.strokeStyle = "rgba(30,30,46,1)";
  ctx.lineWidth   = 1;
  ctx.beginPath();
  ctx.moveTo(PAD.left, PAD.top);
  ctx.lineTo(PAD.left, PAD.top + chartH);
  ctx.lineTo(W - PAD.right, PAD.top + chartH);
  ctx.stroke();
}

// ── Component ──────────────────────────────────────────────────────────────

export default function LayerDeviationPlot({
  sessionId,
  incorrectRecords,
  onRecordSelect,
  deviationData,
  loading,
}: Props) {
  const canvasRef   = useRef<HTMLCanvasElement>(null);
  const [search,    setSearch]    = useState("");
  const [selected,  setSelected]  = useState<string>("");
  const [tooltip,   setTooltip]   = useState<{
    x: number; y: number;
    layer: string;
    trueVal: number | null;
    predVal: number | null;
  } | null>(null);

  // Filtered list for the dropdown
  const filtered = incorrectRecords.filter(r =>
    search === "" ||
    r.id.toLowerCase().includes(search.toLowerCase()) ||
    String(r.label).includes(search) ||
    String(r.predicted).includes(search)
  );

  function handleSelect(id: string) {
    setSelected(id);
    onRecordSelect(id);
  }

  // Redraw chart whenever data changes
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !deviationData) return;
    drawChart(canvas, deviationData);
  }, [deviationData]);

  // ── Tooltip on hover ─────────────────────────────────────────────────────
  function handleMouseMove(e: React.MouseEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas || !deviationData) return;

    const rect   = canvas.getBoundingClientRect();
    const mx     = (e.clientX - rect.left) * (canvas.width  / rect.width);
    const n      = deviationData.layer_names.length;
    const chartW = canvas.width - PAD.left - PAD.right;

    // Find nearest layer index
    let closest = 0;
    let closestDist = Infinity;
    for (let i = 0; i < n; i++) {
      const x = PAD.left + (i / (n - 1)) * chartW;
      const d = Math.abs(mx - x);
      if (d < closestDist) { closestDist = d; closest = i; }
    }

    if (closestDist < 20) {
      setTooltip({
        x: e.clientX,
        y: e.clientY,
        layer:   deviationData.layer_names[closest],
        trueVal: deviationData.true_label_deviations[closest],
        predVal: deviationData.predicted_deviations[closest],
      });
    } else {
      setTooltip(null);
    }
  }

  const selectedRecord = incorrectRecords.find(r => r.id === selected);

  return (
    <div className="panel">
      <h3 className="panel-title">
        <span style={{ fontSize: 14 }}>⊘</span> Layer-Wise Deviation from Prototype
      </h3>

      <p className="layer-dev-desc">
        Select an incorrectly classified record to compare its per-layer
        activation vectors against the <strong>prototype</strong> for its true
        label and its predicted label. The prototype is the mean activation
        vector of all correctly classified records for that label. Deviation is measured in cosine distance.
      </p>

      {/* Record selector */}
      <div className="record-selector">
        <input
          className="text-input"
          placeholder={`Search ${incorrectRecords.length} incorrect records…`}
          value={search}
          onChange={e => setSearch(e.target.value)}
        />
        <div className="record-list">
          {filtered.slice(0, 100).map(r => (
            <button
              key={r.id}
              className={`record-row ${selected === r.id ? "record-row-active" : ""}`}
              onClick={() => handleSelect(r.id)}
            >
              <span className="record-row-id">{r.id}</span>
              <span className="record-row-meta">
                label <strong>{r.label ?? "?"}</strong>
                {" → "}
                predicted <strong style={{ color: "var(--accent2)" }}>{r.predicted}</strong>
              </span>
            </button>
          ))}
          {filtered.length === 0 && (
            <p className="empty-msg" style={{ padding: "10px 12px" }}>No records match.</p>
          )}
          {filtered.length > 100 && (
            <p className="empty-msg" style={{ padding: "6px 12px" }}>
              Showing first 100 — refine search to narrow results.
            </p>
          )}
        </div>
      </div>

      {/* Chart area */}
      {!selected && (
        <div className="layer-dev-empty">
          Select a record above to view its layer-wise deviation.
        </div>
      )}

      {selected && loading && (
        <div className="layer-dev-empty">Computing deviations…</div>
      )}

      {selected && !loading && deviationData && (
        <>
          {/* Legend */}
          <div className="dev-legend">
            <div className="dev-legend-item">
              <span className="dev-legend-dot" style={{ background: COLOR_TRUE }} />
              Deviation from true-label prototype
              {selectedRecord && (
                <span className="dev-legend-label">(label {selectedRecord.label})</span>
              )}
            </div>
            <div className="dev-legend-item">
              <span className="dev-legend-dot" style={{ background: COLOR_PREDICTED }} />
              Deviation from predicted-label prototype
              {selectedRecord && (
                <span className="dev-legend-label">(predicted {selectedRecord.predicted})</span>
              )}
            </div>
          </div>

          {/* Canvas */}
          <div style={{ position: "relative" }}>
            <canvas
              ref={canvasRef}
              width={700}
              height={320}
              style={{
                width: "100%",
                background: "var(--bg)",
                border: "1px solid var(--border)",
                borderRadius: "var(--radius)",
                cursor: "crosshair",
              }}
              onMouseMove={handleMouseMove}
              onMouseLeave={() => setTooltip(null)}
            />

            {tooltip && (
              <div
                style={{
                  position: "fixed",
                  left: tooltip.x + 14,
                  top:  tooltip.y - 10,
                  background: "var(--surface)",
                  border: "1px solid var(--border)",
                  borderRadius: "var(--radius)",
                  padding: "8px 12px",
                  fontSize: 11,
                  fontFamily: "var(--mono)",
                  color: "var(--text)",
                  pointerEvents: "none",
                  zIndex: 100,
                  whiteSpace: "nowrap",
                }}
              >
                <div style={{ color: "var(--muted)", marginBottom: 4 }}>
                  {tooltip.layer}
                </div>
                <div style={{ color: COLOR_TRUE }}>
                  true label:&nbsp;
                  {tooltip.trueVal !== null ? tooltip.trueVal.toFixed(4) : "—"}
                </div>
                <div style={{ color: COLOR_PREDICTED }}>
                  predicted:&nbsp;
                  {tooltip.predVal !== null ? tooltip.predVal.toFixed(4) : "—"}
                </div>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}