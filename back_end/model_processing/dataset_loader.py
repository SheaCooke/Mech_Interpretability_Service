
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from ..common import SUPPORTED_DATASET_EXTENSIONS
import io
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
from pathlib import Path
from ..common import PARQUET_BASE_PATH
from io import BytesIO


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
        df_y = pd.DataFrame(y_test, columns=['val']) 

        df_x.to_parquet(x_path, index=False)
        df_y.to_parquet(y_path, index=False)

    return num_records, x_path, y_path, class_mapping



def _csv_to_parquet(file_bytes: bytes, parquet_path_base: str, label_column: str, session_id: str) -> tuple[int, str, str]:

    csv_content = pd.read_csv(BytesIO(file_bytes))

    x_test = csv_content.drop(columns=[label_column])
    y_test = csv_content[label_column]

    x_path = f'{parquet_path_base}{session_id}\\x_test.parquet'
    y_path = f'{parquet_path_base}{session_id}\\y_test.parquet'

    x_test.to_parquet(x_path, index=False)
    y_test.to_frame().to_parquet(y_path, index=False)

    return len(x_test), x_path, y_path, None
