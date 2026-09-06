import type { ModelData, InferenceSummary, SimilarPair, PredictionFilter, Label, RecordBudget } from "../types";

const API_BASE = "http://localhost:8000";

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? res.statusText);
  }
  return res.json();
}

export async function uploadModel(
  file: File
): Promise<{ session_id: string; model_data: ModelData }> {
  const fd = new FormData();
  fd.append("file", file);
  return apiFetch("/upload/model", { method: "POST", body: fd });
}

export async function uploadDataset(
  sessionId: string,
  file: File,
  labelColumn?: string
): Promise<{ filename: string; num_records: number }> {
  const fd = new FormData();
  fd.append("file", file);
  const params = new URLSearchParams({ session_id: sessionId });
  if (labelColumn) params.set("label_column", labelColumn);
  return apiFetch(`/upload/dataset?${params}`, { method: "POST", body: fd });
}

export async function runInference(
  sessionId: string,
  limit: number,
  maxMemoryMb?: number
): Promise<{
  summary: InferenceSummary;
  num_results: number;
  limit_applied: number;
  limit_requested: number;
  memory_capped: boolean;
  record_budget: RecordBudget;
}> {
  return apiFetch("/inference/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(
      { 
        session_id: sessionId,
        limit: limit,
        max_memory_mb: maxMemoryMb && maxMemoryMb > 0 ? maxMemoryMb : null
      }
    ),
  });
}

/**
 * Ask the backend how many records fit under a RAM ceiling, and what the
 * currently selected record count is projected to cost. Cheap: reads model
 * metadata only, so it is safe to call while the user is typing.
 */
export async function fetchRecordBudget(
  sessionId: string,
  maxMemoryMb: number,
  selectedRecords?: number
): Promise<RecordBudget> {
  return apiFetch("/inference/record-budget", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      max_memory_mb: maxMemoryMb > 0 ? maxMemoryMb : null,
      selected_records: selectedRecords ?? null
    }),
  });
}

export async function deleteSession(sessionId: string): Promise<void> {
  await apiFetch(`/session/${sessionId}`, { method: "DELETE" });
}


export interface ClusterPoint {
  id: string;
  x: number;
  y: number;
  label: number | null;
  predicted: number;
  correct: boolean;
}

export interface ClusterPlotData {
  method: string;
  points: ClusterPoint[];
}

export async function fetchSimilarPairs(
  sessionId: string,
  thresholdLow: number,
  thresholdHigh: number,
  filter: PredictionFilter
): Promise<{ pairs: SimilarPair[]; num_pairs: number }> {
  return apiFetch("/analysis/similar-pairs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ 
      session_id: sessionId,
      threshold_low:  thresholdLow,
      threshold_high: thresholdHigh,
      filter 
    }),
  });
}

export async function fetchClusterPlot(
  sessionId: string,
  filter: PredictionFilter 
): Promise<ClusterPlotData> {
  return apiFetch("/analysis/cluster-plot", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, filter }),
  });
}

 
export interface IncorrectRecord {
  id: string;
  label: Label | null;
  predicted: number;
}
 
export interface LayerDeviationData {
  record_id: string;
  true_label: Label | null;
  predicted_label: number;
  layer_names: string[];
  true_label_deviations: (number | null)[];
  predicted_deviations: (number | null)[];
}
 
export async function fetchIncorrectRecords(
  sessionId: string
): Promise<{ records: IncorrectRecord[]; total: number }> {
  return apiFetch(`/analysis/incorrect-records?session_id=${sessionId}`);
}
 
export async function fetchLayerDeviation(
  sessionId: string,
  recordId: string
): Promise<LayerDeviationData> {
  return apiFetch("/analysis/layer-deviation", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, record_id: recordId }),
  });
}