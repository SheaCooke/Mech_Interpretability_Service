import DropZone from "./DropZone";
import RangeSlider from "./RangeSlider";
import type { Step, DatasetMeta, PredictionFilter  } from "../types";

interface Props {
  step: Step;
  loading: boolean;
  sessionId: string | null;
  datasetMeta: DatasetMeta | null;
  labelColumn: string;
  inferenceLimit: number;
  thresholdLow: number;
  thresholdHigh: number;
  predictionFilter: PredictionFilter;
  onLabelColumnChange: (val: string) => void;
  onInferenceLimitChange: (val: number) => void;
  onThresholdChange: (low: number, high: number) => void;
  onPredictionFilterChange: (val: PredictionFilter) => void;
  onModelFile: (file: File) => void;
  onDatasetFile: (file: File) => void;
  onRunInference: () => void;
  onFindPairs: () => void;
  onClusterPlot: () => void;
}

const stepIdx: Record<Step, number> = {
  "upload-model":   0,
  "upload-dataset": 1,
  "inference":      2,
  "analysis":       3,
};

export default function Sidebar({
  step, loading, sessionId, datasetMeta,
  labelColumn, inferenceLimit, thresholdLow, thresholdHigh, predictionFilter,
  onLabelColumnChange, onInferenceLimitChange, onThresholdChange, onPredictionFilterChange,
  onModelFile, onDatasetFile,
  onRunInference, onFindPairs, onClusterPlot,
}: Props) {
  const idx = stepIdx[step];
  const analysisLocked = idx < 3;
  const totalRecords   = datasetMeta?.num_records ?? 0;

  return (
    <div className="col-left">

      {/* Step 1 – Model */}
      <section className="card">
        <h2 className="card-title">
          <span className="card-num">01</span> Upload Model
        </h2>
        <DropZone
          label="Drop .keras file"
          accept=".keras"
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

        {/* Record count slider */}
        {totalRecords > 0 && (
          <div className="threshold-row">
            <div className="threshold-label-row">
              <span className="threshold-label">Records to run</span>
              <span className="threshold-val">
                {inferenceLimit === totalRecords
                  ? <span>all <span className="threshold-sep">({totalRecords.toLocaleString()})</span></span>
                  : <span>{inferenceLimit.toLocaleString()} <span className="threshold-sep">/ {totalRecords.toLocaleString()}</span></span>
                }
              </span>
            </div>
            <input
              type="range"
              className="slider"
              min={1}
              max={totalRecords}
              step={1}
              value={inferenceLimit}
              onChange={e => onInferenceLimitChange(parseInt(e.target.value, 10))}
              disabled={step !== "inference" || loading}
            />
            <div className="threshold-axis">
              <span>1</span>
              <span style={{ textAlign: "center" }}>sample</span>
              <span style={{ textAlign: "right" }}>all</span>
            </div>
          </div>
        )}

        <button
          className="btn btn-primary"
          onClick={onRunInference}
          disabled={step !== "inference" || loading}
        >
          {loading && step === "inference" ? "Running…" : "Run Inference"}
        </button>
      </section>

      {/* Step 4a – General Analysis */}
      <section className={`card ${analysisLocked ? "card-locked" : ""}`}>
        <h2 className="card-title">
          <span className="card-num">04</span> General Analysis
        </h2>

        {/* Filter dropdown — applies to all analysis below */}
        <div className="filter-row">
          <label className="filter-label">Prediction filter</label>
          <select
            className="filter-select"
            value={predictionFilter}
            onChange={(e) =>
              onPredictionFilterChange(e.target.value as PredictionFilter)
            }
            disabled={step !== "analysis" || loading}
          >
            <option value="all">All predictions</option>
            <option value="correct">Correct only</option>
            <option value="incorrect">Incorrect only</option>
          </select>
        </div>

        {/* Dual-thumb range slider */}
        <div className="threshold-row">
          <div className="threshold-label-row">
            <span className="threshold-label">Similarity range</span>
            <span className="threshold-val">
              {thresholdLow.toFixed(2)}
              <span className="threshold-sep">–</span>
              {thresholdHigh.toFixed(2)}
            </span>
          </div>
          <RangeSlider
            min={0.00}
            max={2.00}
            step={0.01}
            valueLow={thresholdLow}
            valueHigh={thresholdHigh}
            disabled={analysisLocked || loading}
            onChange={onThresholdChange}
          />
          <div className="threshold-axis">
            <span>0.00 (identical)</span>
            <span>1.00 (orthogonal)</span>
            <span>2.00 (opposite)</span>
          </div>
        </div>

        <button
          className="btn btn-primary"
          onClick={onFindPairs}
          disabled={step !== "analysis" || loading}
        >
          {loading ? "Computing…" : "Find Similar Pairs"}
        </button>

        <button
          className="btn btn-primary"
          onClick={onClusterPlot}
          disabled={step !== "analysis" || loading}
          style={{ marginTop: 4 }}
        >
          {loading ? "Computing…" : "Generate Cluster Plot"}
        </button>
      </section>

      {/* Step 4b – Layer-wise Analysis */}
      <section className={`card ${analysisLocked ? "card-locked" : ""}`}>
        <h2 className="card-title">
          <span className="card-num">04b</span> Layer-Wise Analysis
        </h2>
        <p className="sidebar-section-desc">
          Select an incorrectly classified record in the panel on the right to
          compare its per-layer activations against the class prototypes.
        </p>
      </section>

    </div>
  );
}