import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import Sidebar from "../components/Sidebar";
import type { Step } from "../types";

function defaultProps(overrides: Record<string, unknown> = {}) {
  return {
    step: "upload-model" as Step,
    loading: false,
    sessionId: null as string | null,
    datasetMeta: null,
    labelColumn: "",
    threshold: 0.1,
    predictionFilter: "all" as const,
    onLabelColumnChange: vi.fn(),
    onThresholdChange: vi.fn(),
    onPredictionFilterChange: vi.fn(),
    onModelFile: vi.fn(),
    onDatasetFile: vi.fn(),
    onRunInference: vi.fn(),
    onFindPairs: vi.fn(),
    onClusterPlot: vi.fn(),
    ...overrides,
  };
}

describe("<Sidebar />", () => {
  it("renders all four numbered cards", () => {
    render(<Sidebar {...defaultProps()} />);
    expect(screen.getByText(/Upload Model/i)).toBeInTheDocument();
    expect(screen.getByText(/Upload Dataset/i)).toBeInTheDocument();
    expect(screen.getByText(/Run Inference/i)).toBeInTheDocument();
    expect(screen.getByText(/General Analysis/i)).toBeInTheDocument();
  });

  it("disables Run Inference until the inference step is reached", () => {
    render(<Sidebar {...defaultProps({ step: "upload-model" })} />);
    expect(screen.getByRole("button", { name: /Run Inference/i })).toBeDisabled();
  });

  it("enables Run Inference once we are on the inference step", () => {
    render(<Sidebar {...defaultProps({ step: "inference" })} />);
    expect(screen.getByRole("button", { name: /Run Inference/i })).toBeEnabled();
  });

  it("disables analysis controls until the analysis step", () => {
    render(<Sidebar {...defaultProps({ step: "inference" })} />);
    expect(screen.getByRole("button", { name: /Find Similar Pairs/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /Generate Cluster Plot/i })).toBeDisabled();
  });

  it("calls onFindPairs when the user clicks the button on the analysis step", async () => {
    const onFindPairs = vi.fn();
    render(<Sidebar {...defaultProps({ step: "analysis", onFindPairs })} />);
    await userEvent.click(screen.getByRole("button", { name: /Find Similar Pairs/i }));
    expect(onFindPairs).toHaveBeenCalledTimes(1);
  });

  it("propagates label column changes through onLabelColumnChange", async () => {
    const onLabelColumnChange = vi.fn();
    render(
      <Sidebar
        {...defaultProps({ step: "upload-dataset", onLabelColumnChange })}
      />
    );
    const input = screen.getByPlaceholderText(/Label column name/i);
    await userEvent.type(input, "y");
    expect(onLabelColumnChange).toHaveBeenLastCalledWith("y");
  });

  it("propagates filter changes through onPredictionFilterChange", async () => {
    const onPredictionFilterChange = vi.fn();
    render(
      <Sidebar
        {...defaultProps({ step: "analysis", onPredictionFilterChange })}
      />
    );
    const select = screen.getByRole("combobox");
    await userEvent.selectOptions(select, "incorrect");
    expect(onPredictionFilterChange).toHaveBeenCalledWith("incorrect");
  });

  it("shows the dataset metadata once the dataset has been uploaded", () => {
    render(
      <Sidebar
        {...defaultProps({
          step: "inference",
          datasetMeta: { filename: "data.csv", num_records: 1024 },
        })}
      />
    );
    expect(screen.getByText(/data\.csv/)).toBeInTheDocument();
    expect(screen.getByText(/1[.,\s]?024 records/)).toBeInTheDocument();
  });
});
