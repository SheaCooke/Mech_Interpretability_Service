import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import ModelInfo from "../components/ModelInfo";
import type { ModelData } from "../types";

const baseModel: ModelData = {
  format: "keras",
  total_params: 1_234_567,
  trainable_params: 1_000_000,
  num_layers: 3,
  input_shape: [-1, 28, 28],
  output_shape: [-1, 10],
  layers: [
    {
      name: "dense_1",
      type: "Dense",
      activation: "relu",
      num_neurons: 64,
      weight_shape: [784, 64],
      relevant_inference: true,
    },
    {
      name: "dropout_1",
      type: "Dropout",
      activation: null,
      num_neurons: null,
      weight_shape: null,
      relevant_inference: false,
    },
    {
      name: "output",
      type: "Dense",
      activation: "softmax",
      num_neurons: 10,
      weight_shape: [64, 10],
      relevant_inference: true,
    },
  ],
};

describe("<ModelInfo />", () => {
  it("renders the model format in uppercase", () => {
    render(<ModelInfo data={baseModel} />);
    expect(screen.getByText("KERAS")).toBeInTheDocument();
  });

  it("renders total params with locale-aware separators", () => {
    render(<ModelInfo data={baseModel} />);
    // 1,234,567 in en-US locale; we just check that *some* separator appears
    expect(screen.getByText(/1[.,\s]?234[.,\s]?567/)).toBeInTheDocument();
  });

  it("renders the input and output shapes as bracketed lists", () => {
    render(<ModelInfo data={baseModel} />);
    expect(screen.getByText("[-1, 28, 28]")).toBeInTheDocument();
    expect(screen.getByText("[-1, 10]")).toBeInTheDocument();
  });

  it("renders one row per layer with name and type", () => {
    render(<ModelInfo data={baseModel} />);
    expect(screen.getByText("dense_1")).toBeInTheDocument();
    expect(screen.getByText("dropout_1")).toBeInTheDocument();
    expect(screen.getByText("output")).toBeInTheDocument();
  });

  it("marks irrelevant layers with the 'skip' tag", () => {
    render(<ModelInfo data={baseModel} />);
    expect(screen.getByText("skip")).toBeInTheDocument();
  });

  it("shows the activation tag for layers that have one", () => {
    render(<ModelInfo data={baseModel} />);
    expect(screen.getByText("relu")).toBeInTheDocument();
    expect(screen.getByText("softmax")).toBeInTheDocument();
  });

  it("shows the neuron count tag for layers that have one", () => {
    render(<ModelInfo data={baseModel} />);
    expect(screen.getByText("64n")).toBeInTheDocument();
    expect(screen.getByText("10n")).toBeInTheDocument();
  });
});
