import type { ModelData, InferenceSummary, SimilarPair, PredictionFilter } from "../types";

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
  sessionId: string
): Promise<{ summary: InferenceSummary }> {
  return apiFetch("/inference/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
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
  threshold: number,
  filter: PredictionFilter  // ← add this
): Promise<{ pairs: SimilarPair[]; num_pairs: number }> {
  return apiFetch("/analysis/similar-pairs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, threshold, filter }),
  });
}

export async function fetchClusterPlot(
  sessionId: string,
  filter: PredictionFilter  // ← add this
): Promise<ClusterPlotData> {
  return apiFetch("/analysis/cluster-plot", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, filter }),
  });
}