"""Shared pytest fixtures for the Mech_Interpretability_Service test suite.

These fixtures provide deterministic data so that we can test the analysis
classes (Vector_Analyzer, Model_Processor pipeline) without depending on a
particular trained model or dataset.
"""

from __future__ import annotations

import os
import sys

# Make the project root importable so tests can `from analyzer import ...`
# regardless of where pytest is launched from.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pytest


@pytest.fixture
def simple_inference_results():
    """Hand-crafted inference results that exercise Vector_Analyzer's logic.

    Records 0 and 1 are nearly identical (low cosine distance).
    Record 2 is orthogonal (high cosine distance).
    Record 3 has no label so we can verify the no-labels code path.
    """
    return [
        {
            "id": "rec_0",
            "input": [0.0, 1.0],
            "label": 1,
            "predicted": 1,
            "correct": True,
            "activations": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        },
        {
            "id": "rec_1",
            "input": [0.01, 0.99],
            "label": 1,
            "predicted": 1,
            "correct": True,
            "activations": np.array([0.99, 0.01, 0.0, 0.0], dtype=np.float32),
        },
        {
            "id": "rec_2",
            "input": [1.0, 0.0],
            "label": 0,
            "predicted": 0,
            "correct": True,
            "activations": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        },
        {
            "id": "rec_3",
            "input": [0.5, 0.5],
            "label": None,
            "predicted": 1,
            "correct": False,
            "activations": np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
        },
    ]


@pytest.fixture
def csv_dataset(tmp_path):
    """Write a small CSV dataset to a temp file and return its path."""
    path = tmp_path / "tiny.csv"
    path.write_text(
        "f1,f2,f3,label\n"
        "0,0,0,0\n"
        "0,1,0,1\n"
        "1,0,1,0\n"
        "1,1,1,1\n"
    )
    return path


@pytest.fixture
def npz_dataset(tmp_path):
    """Write a small NPZ dataset (with labels) and return its path."""
    path = tmp_path / "tiny.npz"
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0])
    np.savez(path, x_test=x, y_test=y)
    return path


@pytest.fixture
def npz_dataset_no_labels(tmp_path):
    """Write a small NPZ dataset (no labels)."""
    path = tmp_path / "tiny_unlabeled.npz"
    x = np.array([[0, 0], [1, 1]], dtype=np.float32)
    np.savez(path, x_test=x)
    return path


def _project_test_asset(name: str) -> str:
    """Return the absolute path to a file in the repo's `test/` folder, or ''."""
    candidate = os.path.join(PROJECT_ROOT, "test", name)
    return candidate if os.path.exists(candidate) else ""


@pytest.fixture
def keras_model_path():
    """Path to the keras model checked into the repo, or skip the test."""
    path = _project_test_asset("test_model.keras")
    if not path:
        pytest.skip("test/test_model.keras not present; skipping integration test")
    return path


@pytest.fixture
def keras_data_path():
    """Path to the npz data checked into the repo, or skip the test."""
    path = _project_test_asset("test_data.npz")
    if not path:
        pytest.skip("test/test_data.npz not present; skipping integration test")
    return path
