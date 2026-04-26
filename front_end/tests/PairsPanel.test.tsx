import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import PairsPanel from "../components/PairsPanel";
import type { SimilarPair } from "../types";

function makePair(overrides: Partial<SimilarPair> = {}): SimilarPair {
  return {
    id_a: "rec_a",
    id_b: "rec_b",
    distance: 0.0123,
    label_a: 1,
    label_b: 1,
    ...overrides,
  };
}

describe("<PairsPanel />", () => {
  it("shows the empty state when there are no pairs", () => {
    render(<PairsPanel pairs={[]} />);
    expect(screen.getByText(/No pairs found/i)).toBeInTheDocument();
  });

  it("renders pairs and includes the count in the title", () => {
    const pairs = [makePair(), makePair({ id_a: "x", id_b: "y", distance: 0.5 })];
    render(<PairsPanel pairs={pairs} />);
    expect(screen.getByText(/Similar Activation Pairs \(2\)/)).toBeInTheDocument();
    expect(screen.getByText("rec_a")).toBeInTheDocument();
    expect(screen.getByText("x")).toBeInTheDocument();
  });

  it("formats the distance to 4 decimal places", () => {
    render(<PairsPanel pairs={[makePair({ distance: 0.123456789 })]} />);
    expect(screen.getByText("0.1235")).toBeInTheDocument();
  });

  it("renders an em-dash for null labels", () => {
    render(
      <PairsPanel
        pairs={[makePair({ label_a: null, label_b: null })]}
      />
    );
    // there should be at least two em dashes (one for each label cell)
    expect(screen.getAllByText("—").length).toBeGreaterThanOrEqual(2);
  });

  it("caps the visible rows at 200 and shows an overflow notice", () => {
    const pairs: SimilarPair[] = Array.from({ length: 250 }, (_, i) =>
      makePair({ id_a: `a_${i}`, id_b: `b_${i}`, distance: i / 1000 })
    );
    render(<PairsPanel pairs={pairs} />);
    // First row shown
    expect(screen.getByText("a_0")).toBeInTheDocument();
    // Row 200 should not be rendered
    expect(screen.queryByText("a_200")).not.toBeInTheDocument();
    expect(screen.getByText(/Showing first 200 of 250 pairs/i)).toBeInTheDocument();
  });
});
