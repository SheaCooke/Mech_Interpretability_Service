import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import StatusBar from "../components/StatusBar";

describe("<StatusBar />", () => {
  it("renders the message text", () => {
    render(<StatusBar msg="Loading…" type="info" />);
    expect(screen.getByText("Loading…")).toBeInTheDocument();
  });

  it.each([
    ["info", "status-info"],
    ["error", "status-error"],
    ["success", "status-success"],
  ] as const)("applies status-%s class for type=%s", (type, expectedClass) => {
    const { container } = render(<StatusBar msg="x" type={type} />);
    expect(container.firstChild).toHaveClass("status-bar");
    expect(container.firstChild).toHaveClass(expectedClass);
  });
});
