
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from ..common import SUPPORTED_DATASET_EXTENSIONS
from .types import DataRecord
import io



def load_dataset(
    file_bytes: bytes,
    ext: str,
    label_column: Optional[str] = None
) -> list[DataRecord]:

    if ext not in SUPPORTED_DATASET_EXTENSIONS:
        raise ValueError(
            f"Unsupported dataset format: .{ext}. "
            f"Supported formats: {SUPPORTED_DATASET_EXTENSIONS}"
        )

    loaders = {
        'csv': _load_csv,
        'npz': _load_npz,
    }
    return loaders[ext](file_bytes, label_column)


def _load_csv(file_bytes: bytes, label_column: Optional[str]) -> list[DataRecord]:
    df = pd.read_csv(io.BytesIO(file_bytes))

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

        # Preserve the native Python type of the label.
        # pandas infers int, float, or str depending on the column content.
        # Converting to int() would crash on string labels like "setosa".
        if label is not None:
            raw = label.item() if hasattr(label, 'item') else label
            # Keep as int if it is a whole number float (e.g. 1.0 → 1),
            # otherwise preserve str or non-whole float as-is
            if isinstance(raw, float) and raw.is_integer():
                native_label = int(raw)
            else:
                native_label = raw
        else:
            native_label = None

        records.append(DataRecord(
            id    = f"record_{idx}",
            input = tuple(input_data.flatten().tolist()),
            label = native_label,
        ))

    return records


def _load_npz(file_bytes: bytes, label_column: Optional[str] = None) -> list[DataRecord]:
    with np.load(io.BytesIO(file_bytes)) as data:
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
            # .item() converts numpy scalar to native Python type:
            # np.int64 → int, np.float64 → float, np.str_ → str
            # handles integer, float, and string label arrays.
            label = y_test[idx].item() if y_test is not None else None,
        )
        for idx, _ in enumerate(x_test)
    ]


def _array_to_nested_tuple(arr: np.ndarray):
    """Recursively convert a numpy array to nested tuples."""
    if arr.ndim == 1:
        return tuple(arr.tolist())
    return tuple(_array_to_nested_tuple(row) for row in arr)