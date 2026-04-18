import DropZone from "./DropZone";
import type { Step, DatasetMeta } from "../types";

interface Props {
  step: Step;
  loading: boolean;
  sessionId: string | null;
  datasetMeta: DatasetMeta | null;
  labelColumn: string;
  threshold: number;
  onLabelColumnChange: (val: string) => void;
  onThresholdChange: (val: number) => void;
  onModelFile: (file: File) => void;
  onDatasetFile: (file: File) => void;
  onRunInference: () => void;
  onFindPairs: () => void;
}

const stepIdx: Record<Step, number> = {
  "upload-model":   0,
  "upload-dataset": 1,
  "inference":      2,
  "analysis":       3,
};

export default function Sidebar({
  step, loading, sessionId, datasetMeta,
  labelColumn, threshold,
  onLabelColumnChange, onThresholdChange,
  onModelFile, onDatasetFile,
  onRunInference, onFindPairs,
}: Props) {
  const idx = stepIdx[step];

  return (
    <div className="col-left">

      {/* Step 1 – Model */}
      <section className="card">
        <h2 className="card-title">
          <span className="card-num">01</span> Upload Model
        </h2>
        <DropZone
          label="Drop .keras / .onnx / .pt / .pth"
          accept=".keras,.onnx,.pt,.pth"
          onFile={onModelFile}
          disabled={loading || !!sessionId}
        />
      </section>

      {/* Step 2 – Dataset */}
      <section className={`card ${step === "upload-model" ? "card-locked" : ""}`}>
        <h2 className="card-title">
          <span className="card-num">02</span> Upload Dataset
        </h2>
        <input
          className="text-input"
          placeholder="Label column name (optional)"
          value={labelColumn}
          onChange={(e) => onLabelColumnChange(e.target.value)}
          disabled={step === "upload-model" || loading}
        />
        <DropZone
          label="Drop .csv or .npz"
          accept=".csv,.npz"
          onFile={onDatasetFile}
          disabled={step === "upload-model" || loading || !!datasetMeta}
        />
        {datasetMeta && (
          <p className="meta-line">
            {datasetMeta.filename} &mdash;{" "}
            {datasetMeta.num_records.toLocaleString()} records
          </p>
        )}
      </section>

      {/* Step 3 – Inference */}
      <section className={`card ${idx < 2 ? "card-locked" : ""}`}>
        <h2 className="card-title">
          <span className="card-num">03</span> Run Inference
        </h2>
        <button
          className="btn btn-primary"
          onClick={onRunInference}
          disabled={step !== "inference" || loading}
        >
          {loading && step === "inference" ? "Running…" : "Run Inference"}
        </button>
      </section>

      {/* Step 4 – Analysis */}
      <section className={`card ${idx < 3 ? "card-locked" : ""}`}>
        <h2 className="card-title">
          <span className="card-num">04</span> Similar Activations
        </h2>
        <div className="threshold-row">
          <label className="threshold-label">
            Threshold{" "}
            <span className="threshold-val">{threshold.toFixed(2)}</span>
          </label>
          <input
            type="range" min={0.01} max={0.5} step={0.01}
            value={threshold}
            onChange={(e) => onThresholdChange(parseFloat(e.target.value))}
            disabled={step !== "analysis" || loading}
            className="slider"
          />
        </div>
        <button
          className="btn btn-primary"
          onClick={onFindPairs}
          disabled={step !== "analysis" || loading}
        >
          {loading && step === "analysis" ? "Computing…" : "Find Similar Pairs"}
        </button>
      </section>

    </div>
  );
}