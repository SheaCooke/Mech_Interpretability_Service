export interface LayerData {
  name: string;
  type: string;
  activation: string | null;
  num_neurons: number | null;
  weight_shape: number[] | null;
  relevant_inference: boolean;
}

export interface ModelData {
  format: string;
  total_params: number;
  trainable_params: number;
  num_layers: number;
  input_shape: number[];
  output_shape: number[];
  layers: LayerData[];
}

export interface InferenceSummary {
  total_records: number;
  has_labels: boolean;
  correct?: number;
  incorrect?: number;
  accuracy?: number;
  per_class_accuracy?: Record<string, { total: number; correct: number; accuracy: number }>;
}

export interface InferenceResult {
  id: string;
  label: number | null;
  predicted: number;
  correct: boolean;
}

export interface SimilarPair {
  id_a: string;
  id_b: string;
  distance: number;
  label_a: number | null;
  label_b: number | null;
}

export interface DatasetMeta {
  filename: string;
  num_records: number;
}

export interface StatusMessage {
  msg: string;
  type: "info" | "error" | "success";
}

export type Step = "upload-model" | "upload-dataset" | "inference" | "analysis";

export const STEP_ORDER: Step[] = [
  "upload-model",
  "upload-dataset",
  "inference",
  "analysis",
];

export const STEP_LABELS: Record<Step, string> = {
  "upload-model":   "Model",
  "upload-dataset": "Dataset",
  "inference":      "Inference",
  "analysis":       "Analysis",
};