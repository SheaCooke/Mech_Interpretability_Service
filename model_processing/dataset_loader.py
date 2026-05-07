"""
dataset_loader.py

Responsible solely for loading dataset files and converting them into
immutable DataRecord objects. Kept separate from model loading and
inference so that each class has a single responsibility.

Supported formats: .csv, .npz
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd

from .types import DataRecord


SUPPORTED_FORMATS = ['csv', 'npz']


def load_dataset(
    file_path:    str,
    label_column: Optional[str] = None,
) -> list[DataRecord]:
    """
    Load a dataset file and return a list of immutable DataRecord objects.

    Args:
        file_path:    path to a .csv or .npz file
        label_column: column name for labels in CSV files (ignored for NPZ)

    Raises:
        ValueError: if the format is unsupported or the label column is missing
    """
    ext = file_path.rsplit('.', 1)[-1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported dataset format: .{ext}. "
            f"Supported formats: {SUPPORTED_FORMATS}"
        )

    loaders = {
        'csv': _load_csv,
        'npz': _load_npz,
    }
    return loaders[ext](file_path, label_column)


def _load_csv(file_path: str, label_column: Optional[str]) -> list[DataRecord]:
    df = pd.read_csv(file_path)

    if label_column and label_column not in df.columns:
        raise ValueError(
            f"Label column '{label_column}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )

    records = []
    for idx, row in df.iterrows():
        if label_column:
            label      = row[label_column]
            input_data = row.drop(label_column).to_numpy().astype(np.float32)
        else:
            label      = None
            input_data = row.to_numpy().astype(np.float32)

        records.append(DataRecord(
            id    = f"record_{idx}",
            input = tuple(input_data.flatten().tolist()),
            label = int(label) if label is not None else None,
        ))

    return records


def _load_npz(file_path: str, label_column: Optional[str] = None) -> list[DataRecord]:
    data = np.load(file_path, allow_pickle=False)

    if 'x_test' not in data:
        raise ValueError(
            f"NPZ file must contain 'x_test' array. "
            f"Found keys: {list(data.keys())}"
        )

    x_test = data['x_test']
    y_test = data['y_test'] if 'y_test' in data else None

    return [
        DataRecord(
            id    = f"record_{idx}",
            # Store as nested tuples so the DataRecord remains hashable.
            # np.ndarray is not hashable; tuple is.
            input = _array_to_nested_tuple(x_test[idx].astype(np.float32)),
            label = int(y_test[idx]) if y_test is not None else None,
        )
        for idx, _ in enumerate(x_test)
    ]


def _array_to_nested_tuple(arr: np.ndarray):
    """Recursively convert a numpy array to nested tuples."""
    if arr.ndim == 1:
        return tuple(arr.tolist())
    return tuple(_array_to_nested_tuple(row) for row in arr)