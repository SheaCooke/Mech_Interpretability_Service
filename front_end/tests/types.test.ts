import { describe, it, expect } from "vitest";
import {
  STEP_ORDER,
  STEP_LABELS,
  type Step,
} from "../types";

describe("STEP_ORDER", () => {
  it("contains every step exactly once in pipeline order", () => {
    expect(STEP_ORDER).toEqual([
      "upload-model",
      "upload-dataset",
      "inference",
      "analysis",
    ]);
  });

  it("has no duplicates", () => {
    const set = new Set<Step>(STEP_ORDER);
    expect(set.size).toBe(STEP_ORDER.length);
  });
});

describe("STEP_LABELS", () => {
  it("provides a human-readable label for every Step", () => {
    for (const step of STEP_ORDER) {
      expect(STEP_LABELS[step]).toBeTruthy();
      expect(typeof STEP_LABELS[step]).toBe("string");
    }
  });

  it("maps known steps to their UI labels", () => {
    expect(STEP_LABELS["upload-model"]).toBe("Model");
    expect(STEP_LABELS["upload-dataset"]).toBe("Dataset");
    expect(STEP_LABELS["inference"]).toBe("Inference");
    expect(STEP_LABELS["analysis"]).toBe("Analysis");
  });
});
