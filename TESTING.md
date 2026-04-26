# Testing

This repo has two independent test suites: one for the Python backend and one for the TypeScript front_end. Each is run from its own directory.

## Python tests

The Python suite lives in `tests/` at the repo root and is configured by `pytest.ini`. It covers `analyzer.py`, `model_processing/model_processor.py`, and `model_processing/vector_analyzer.py`. The `Vector_Analyzer` and summarisation tests are pure logic and run quickly. The `Model_Processor` integration tests (model loading, full inference) reuse the assets in the `test/` directory (`test_model.keras`, `test_data.npz`); they auto-skip if those files are absent.

### Setup

From the repo root, with the existing virtual environment activated:

```powershell
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
pip install pytest
```

```bash
# macOS/Linux
source venv/bin/activate
pip install pytest
```

The other dependencies (`numpy`, `scipy`, `keras`, `tensorflow`, `torch`, `onnx`, `pandas`, `umap-learn`, `scikit-learn`) are already in `requirements.txt`.

### Running

```bash
# Full suite
pytest

# Just one file
pytest tests/test_vector_analyzer.py

# Just one test
pytest tests/test_vector_analyzer.py::TestFindAllSimilarPairs::test_pairs_are_sorted_by_distance_ascending

# With verbose output
pytest -v

# With a coverage report (after `pip install pytest-cov`)
pytest --cov=. --cov-report=term-missing --cov-report=html
```

### What's covered

`tests/test_analyzer.py` checks that the placeholder `Analyzer` class is importable and instantiable. `tests/test_vector_analyzer.py` builds synthetic inference results to verify the id mapping, activation matrix, distance matrix (square, symmetric, zero diagonal), and `find_all_similar_pairs` (upper-triangle dedup, ascending distance order, threshold filtering, label propagation). `tests/test_model_processor.py` covers format detection, CSV/NPZ dataset loading (with and without labels, error paths for missing columns and unsupported extensions), the private `__summarize_results` (no-labels short circuit, overall and per-class accuracy), plus integration tests that load the real Keras model and run both the unbatched and batched inference paths to make sure they agree.

## TypeScript tests

The front_end suite uses Vitest with React Testing Library and jsdom. Tests live in `front_end/tests/`. Configuration is in `front_end/vitest.config.ts`; the setup file `front_end/tests/setup.ts` registers `jest-dom` matchers and runs cleanup between tests.

### Setup

```bash
cd front_end
npm install
```

This will install the new dev dependencies that were added to `package.json` (`vitest`, `jsdom`, `@testing-library/react`, `@testing-library/jest-dom`, `@testing-library/user-event`, `@vitest/coverage-v8`).

### Running

```bash
cd front_end

# Single run, exits with status code (good for CI)
npm test

# Watch mode — re-runs tests as you edit
npm run test:watch

# Vitest UI (browser dashboard)
npm run test:ui

# With a coverage report
npm run test:coverage
```

To run just one file: `npx vitest run tests/Header.test.tsx`. To run by name: `npx vitest run -t "calls onReset"`.

### What's covered

`tests/types.test.ts` locks down `STEP_ORDER` and `STEP_LABELS` so the pipeline's UI labels can't drift silently. `tests/StatusBar.test.tsx`, `tests/ModelInfo.test.tsx`, `tests/SummaryPanel.test.tsx`, and `tests/PairsPanel.test.tsx` are render-and-assert tests for the data-display components — they verify formatting (locale numbers, percentages, four-decimal distances), conditional sections (per-class accuracy block, empty states), the 200-row pairs cap, and tag rendering for layers. `tests/DropZone.test.tsx` exercises the file picker and drag-and-drop paths, plus the `disabled` prop. `tests/Header.test.tsx` and `tests/Sidebar.test.tsx` use `@testing-library/user-event` to verify the wiring between buttons/selects/inputs and the callback props (navigation, reset, find-pairs, label column, prediction filter). `tests/client.test.ts` stubs `globalThis.fetch` to verify every API call's URL, method, headers, and body shape, plus the JSON-detail and statusText error fallbacks.

## Running both suites together

A simple one-liner from the repo root:

```bash
# macOS/Linux
pytest && (cd front_end && npm test)
```

```powershell
# Windows (PowerShell)
pytest; if ($LASTEXITCODE -eq 0) { Push-Location front_end; npm test; Pop-Location }
```
