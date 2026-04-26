import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// Vitest configuration kept separate from vite.config.ts so the dev server
// config doesn't have to know anything about tests.
export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./tests/setup.ts"],
    css: false,
    include: ["tests/**/*.test.{ts,tsx}"],
    coverage: {
      reporter: ["text", "html"],
      include: ["**/*.{ts,tsx}"],
      exclude: [
        "node_modules/**",
        "tests/**",
        "**/*.d.ts",
        "vite.config.ts",
        "vitest.config.ts",
      ],
    },
  },
});
