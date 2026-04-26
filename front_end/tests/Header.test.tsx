import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import Header from "../components/Header";

function defaultProps(overrides = {}) {
  return {
    step: "upload-model" as const,
    sessionId: null as string | null,
    currentPage: "home" as const,
    onNavigate: vi.fn(),
    onReset: vi.fn(),
    ...overrides,
  };
}

describe("<Header />", () => {
  it("renders the four pipeline steps when on the home page", () => {
    render(<Header {...defaultProps()} />);
    expect(screen.getByText("Model")).toBeInTheDocument();
    expect(screen.getByText("Dataset")).toBeInTheDocument();
    expect(screen.getByText("Inference")).toBeInTheDocument();
    expect(screen.getByText("Analysis")).toBeInTheDocument();
  });

  it("hides the step nav when not on the home page", () => {
    render(<Header {...defaultProps({ currentPage: "instructions" })} />);
    // The page nav button "Instructions" still exists, but the step pill labels
    // (single-word "Model"/"Dataset"/"Inference"/"Analysis") shouldn't be shown.
    expect(screen.queryByText("Model")).not.toBeInTheDocument();
  });

  it("hides the Reset button when there is no active session", () => {
    render(<Header {...defaultProps()} />);
    expect(screen.queryByRole("button", { name: /Reset/i })).not.toBeInTheDocument();
  });

  it("shows the Reset button when a session is active and on the home page", () => {
    render(<Header {...defaultProps({ sessionId: "abc-123" })} />);
    expect(screen.getByRole("button", { name: /Reset/i })).toBeInTheDocument();
  });

  it("calls onNavigate when a nav button is clicked", async () => {
    const onNavigate = vi.fn();
    render(<Header {...defaultProps({ onNavigate })} />);

    await userEvent.click(screen.getByRole("button", { name: /Instructions/i }));
    expect(onNavigate).toHaveBeenCalledWith("instructions");
  });

  it("calls onReset when the Reset button is clicked", async () => {
    const onReset = vi.fn();
    render(<Header {...defaultProps({ sessionId: "abc-123", onReset })} />);

    await userEvent.click(screen.getByRole("button", { name: /Reset/i }));
    expect(onReset).toHaveBeenCalledTimes(1);
  });
});
