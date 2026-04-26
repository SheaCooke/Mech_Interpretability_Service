import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import SummaryPanel from "../components/SummaryPanel";
import type { InferenceSummary } from "../types";

describe("<SummaryPanel />", () => {
  it("shows just the record count when labels are missing", () => {
    const summary: InferenceSummary = {
      total_records: 42,
      has_labels: false,
    };
    render(<SummaryPanel summary={summary} />);
    expect(screen.getByText("Records")).toBeInTheDocument();
    expect(screen.getByText("42")).toBeInTheDocument();
    expect(screen.queryByText("Accuracy")).not.toBeInTheDocument();
  });

  it("shows accuracy/correct/incorrect when has_labels is true", () => {
    const summary: InferenceSummary = {
      total_records: 100,
      has_labels: true,
      correct: 75,
      incorrect: 25,
      accuracy: 0.75,
    };
    render(<SummaryPanel summary={summary} />);
    expect(screen.getByText("75.00%")).toBeInTheDocument();
    expect(screen.getByText("75")).toBeInTheDocument();
    expect(screen.getByText("25")).toBeInTheDocument();
  });

  it("renders one row per class in per_class_accuracy", () => {
    const summary: InferenceSummary = {
      total_records: 4,
      has_labels: true,
      correct: 3,
      incorrect: 1,
      accuracy: 0.75,
      per_class_accuracy: {
        "0": { total: 2, correct: 2, accuracy: 1.0 },
        "1": { total: 2, correct: 1, accuracy: 0.5 },
      },
    };
    render(<SummaryPanel summary={summary} />);
    expect(screen.getByText("0")).toBeInTheDocument();
    expect(screen.getByText("1")).toBeInTheDocument();
    expect(screen.getByText("100%")).toBeInTheDocument();
    expect(screen.getByText("50%")).toBeInTheDocument();
  });
});
