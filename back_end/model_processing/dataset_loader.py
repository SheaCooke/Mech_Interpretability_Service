
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from ..common import SUPPORTED_DATASET_EXTENSIONS
from .types import DataRecord
import io
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
from pathlib import Path
from ..common import PARQUET_BASE_PATH
from io import BytesIO



def load_dataset(
    file_bytes: bytes,
    ext: str,
    label_column: Optional[str] = None
) -> list[DataRecord]: #TODO: should stream to a file instead of returning a value

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


#TODO: add support for test datasets that have a both string and numeric feature types

def _load_csv(file_bytes: bytes, label_column: Optional[str]) -> list[DataRecord]: #TODO: label column is required for csv files
    df = pd.read_csv(io.BytesIO(file_bytes))

    if label_column and label_column not in df.columns:
        raise ValueError(
            f"Label column '{label_column}' not found in csv file."
            f"Available columns: {list(df.columns)}"
        )
    elif not label_column:
        raise ValueError(
            "Label column is required when uploading a csv file. Please enter it before selecting the csv file."
        )

    records = []

    for idx, row in df.iterrows():

        label = row[label_column]
        input_data = row.drop(label_column).to_numpy().astype(np.float32)

        raw = label.item() if hasattr(label, 'item') else label

        if isinstance(raw, float) and raw.is_integer():
            native_label = int(raw)
        else:
            native_label = raw

        records.append(DataRecord(
            id = f"record_{idx}",
            input = tuple(input_data.flatten().tolist()),
            label = native_label
        ))

    return records


def _load_npz(file_bytes: bytes, label_column: Optional[str] = None) -> list[DataRecord]:
    with np.load(io.BytesIO(file_bytes)) as data:
        if 'x_test' not in data or 'y_test' not in data:
            raise ValueError(
                f"NPZ file must contain 'x_test' and 'y_test'. "
                f"Found keys: {list(data.keys())}"
            )

        x_test = data['x_test']
        y_test = data['y_test']


    return [
        DataRecord(
            id    = f"record_{idx}",
            input = _array_to_nested_tuple(x_test[idx].astype(np.float32)),
            label = y_test[idx].item()
        )
        for idx, _ in enumerate(x_test)
    ]


def _array_to_nested_tuple(arr: np.ndarray):
    """Recursively convert a numpy array to nested tuples."""
    if arr.ndim == 1:
        return tuple(arr.tolist())
    return tuple(_array_to_nested_tuple(row) for row in arr)


def convert_to_parquet(file_extension: str, file_bytes: bytes, session_id: str, label_column: Optional[str] = None) -> tuple[int, str, str, list]:

    if file_extension == 'csv':
        if not label_column:
            raise ValueError('A specified label column is required when uploading a csv file.')
        return _csv_to_parquet(file_bytes, PARQUET_BASE_PATH, label_column, session_id)
    elif file_extension == 'npz':
        return _npz_to_parquet(file_bytes, PARQUET_BASE_PATH, session_id)

    return None
    

def _npz_to_parquet(file_bytes: bytes, parquet_path_base: str, session_id: str) -> tuple[int, str, str, list]:

    x_path = f'{parquet_path_base}{session_id}\\x_test.parquet'
    y_path = f'{parquet_path_base}{session_id}\\y_test.parquet'
    num_records = 0

    with np.load(io.BytesIO(file_bytes)) as data:
        if 'x_test' not in data or 'y_test' not in data:
            raise ValueError(
                    f"NPZ file must contain 'x_test' and 'y_test'. "
                    f"Found keys: {list(data.keys())}"
                )
        
        x_test = data['x_test']
        y_test = data['y_test']

        class_mapping = None
        if 'class_names' in data:
            class_mapping = data['class_names'].tolist()

        num_records = len(x_test)

        if x_test.ndim == 3: #3d image data needs to be reshaped before it can be stored
            num_samples = x_test.shape[0]
            x_test = x_test.reshape(num_samples, -1)

        df_x = pd.DataFrame(x_test)
        df_y = pd.DataFrame(y_test)

        df_x.to_parquet(x_path, index=False)
        df_y.to_parquet(y_path, index=False)

    return num_records, x_path, y_test, class_mapping



def _csv_to_parquet(file_bytes: bytes, parquet_path_base: str, label_column: str, session_id: str) -> tuple[int, str, str]:

    csv_content = pd.read_csv(BytesIO(file_bytes))

    x_test = csv_content.drop(columns=[label_column])
    y_test = csv_content[label_column]

    x_path = f'{parquet_path_base}{session_id}\\x_test.parquet'
    y_path = f'{parquet_path_base}{session_id}\\y_test.parquet'

    x_test.to_parquet(x_path, index=False)
    y_test.to_frame().to_parquet(y_path, index=False)

    return len(x_test), x_path, y_path, None
