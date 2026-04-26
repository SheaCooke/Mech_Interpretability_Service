// Test setup file — runs before every test file.
// Adds jest-dom's custom matchers (e.g. toBeInTheDocument) to vitest's expect,
// and resets mocks between tests so a stub from one test can't leak.
import "@testing-library/jest-dom/vitest";
import { afterEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});
