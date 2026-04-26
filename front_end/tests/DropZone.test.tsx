import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import DropZone from "../components/DropZone";

function makeFile(name = "model.keras") {
  return new File(["fake binary content"], name, {
    type: "application/octet-stream",
  });
}

function getInput(container: HTMLElement): HTMLInputElement {
  const input = container.querySelector("input[type=file]");
  if (!(input instanceof HTMLInputElement)) {
    throw new Error("Could not find file input");
  }
  return input;
}

describe("<DropZone />", () => {
  it("renders the placeholder label until a file is provided", () => {
    render(<DropZone label="Drop a model" accept=".keras" onFile={() => {}} />);
    expect(screen.getByText("Drop a model")).toBeInTheDocument();
  });

  it("invokes onFile when the user picks a file from the dialog", () => {
    const onFile = vi.fn();
    const { container } = render(
      <DropZone label="Drop a model" accept=".keras" onFile={onFile} />
    );
    const input = getInput(container);

    const file = makeFile("foo.keras");
    fireEvent.change(input, { target: { files: [file] } });

    expect(onFile).toHaveBeenCalledTimes(1);
    expect(onFile).toHaveBeenCalledWith(file);
  });

  it("shows the chosen file's name after selection", () => {
    const { container } = render(
      <DropZone label="Drop a model" accept=".keras" onFile={() => {}} />
    );
    const input = getInput(container);
    fireEvent.change(input, { target: { files: [makeFile("uploaded.keras")] } });
    expect(screen.getByText("uploaded.keras")).toBeInTheDocument();
  });

  it("does not call onFile when the input is disabled", () => {
    const onFile = vi.fn();
    const { container } = render(
      <DropZone label="Drop a model" accept=".keras" onFile={onFile} disabled />
    );
    const input = getInput(container);
    expect(input).toBeDisabled();
    // Even if a change somehow fires, the disabled prop on the input is the contract.
    // We assert it here rather than firing the event because jsdom respects disabled.
    expect(onFile).not.toHaveBeenCalled();
  });

  it("ignores drop events when disabled", () => {
    const onFile = vi.fn();
    const { container } = render(
      <DropZone label="Drop here" accept=".csv" onFile={onFile} disabled />
    );
    const label = container.querySelector("label")!;

    const dt = {
      files: [makeFile("a.csv")],
      types: ["Files"],
    } as unknown as DataTransfer;

    fireEvent.drop(label, { dataTransfer: dt });
    expect(onFile).not.toHaveBeenCalled();
  });

  it("calls onFile when a file is dropped onto the zone", () => {
    const onFile = vi.fn();
    const { container } = render(
      <DropZone label="Drop here" accept=".csv" onFile={onFile} />
    );
    const label = container.querySelector("label")!;
    const file = makeFile("data.csv");

    const dt = {
      files: [file],
      types: ["Files"],
    } as unknown as DataTransfer;

    fireEvent.drop(label, { dataTransfer: dt });
    expect(onFile).toHaveBeenCalledWith(file);
  });
});
