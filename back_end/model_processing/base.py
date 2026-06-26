
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import pyarrow.parquet as pq

from .types import ModelMetadata, InferenceRecord, DataRecord

logger = logging.getLogger(__name__)

class ModelStrategy(ABC):

    def __init__(self):
        self._class_names: Optional[dict[int, Any]] = None

    @abstractmethod
    def load(self, file_path: str) -> Any:
        """
        Load a model and return the frameworks model object
        """

    @abstractmethod
    def extract_model_data(self, model: Any) -> ModelMetadata:
        """
        Inspect a loaded model and return an immutable ModelMetadata snapshot
        """

    def run_inference(self, model: Any, x_test_path: str, y_test_path: str, class_mapping: Optional[list]) -> list[InferenceRecord]:

        self._build_class_map(class_mapping) 
        self._prepare(model)
        results = []

        x_test = pq.ParquetFile(x_test_path)
        y_test = pq.ParquetFile(y_test_path)

        batches_x = x_test.iter_batches(batch_size=100)
        batches_y = y_test.iter_batches(batch_size=100)
        #TODO: collect unique labels in a set while streaming in the records
        #TODO: matches should be passed to the model using tensorflow lib

        record_id = 1

        for batch_x, batch_y in zip(batches_x, batches_y):
            rows_x = batch_x.to_pylist()
            rows_y = batch_y.to_pylist()

            for rx, ry in zip(rows_x, rows_y):
                tensor           = self._prepare_input(tuple(rx.values())) 
                raw_out, per_layer = self._forward(model, tensor)
                predicted        = self._get_prediction(raw_out)
                inf_record       = self._build_record(record_id, tuple(rx.values()), predicted, per_layer, ry['val'])
                results.append(inf_record)

                record_id += 1
            
            if record_id % 200 == 0:
                logger.info(f'completed inference for record number {record_id}')
        
        self._teardown(model)

        return results

    @abstractmethod
    def _prepare_input(self, raw: tuple) -> Any:
        """
        Convert a raw input tuple into the tensor format the model expects.
        Must handle any necessary shape transformations 
        """

    @abstractmethod
    def _forward(self, model: Any, tensor: Any) -> tuple[Any, dict[str, list[float]]]:
        """
        Run a single forward pass.
        """

    @abstractmethod
    def _prepare(self, model: Any) -> None:
        """
        setup before the inference loop begins.
        """

    @abstractmethod
    def _teardown(self, model: Any) -> None:
        """
        cleanup after the inference loop
        """

    def _get_prediction(self, raw_output: np.ndarray) -> int:
        """
        Extract the predicted class index from the raw model output.
        """
        return int(np.argmax(raw_output))
        
    def _resolve_predicted(self, predicted_idx: int) -> Any:
        """
        handles both string and numeric label values
        """
        if not self._class_names:
            return predicted_idx
        return self._class_names.get(predicted_idx, predicted_idx)

    def _build_class_map(self, labels: Optional[list]) -> None:
        """
        Inspect the dataset labels to determine whether a class name mapping is needed
        """
        # labels = [r.label for r in records if r.label is not None] 
        if not labels:
            self._class_names = {}
            return
 
        if isinstance(labels[0], int):
            # Integer labels, predicted index is the label, no mapping needed
            self._class_names = {}
            return
 
        # string or float labels, sort unique values to reconstruct index order
        # string labels may be handled by a pre-processing layer in the model
        unique = sorted(set(labels), key=lambda x: (str(type(x)), x))
        self._class_names = {idx: val for idx, val in enumerate(unique)}

    def _build_record(self, record_id: int, input_row, predicted: int, per_layer: dict[str, list[float]], label) -> InferenceRecord:
        flat = np.concatenate([
            np.array(v, dtype=np.float32).flatten()
            for v in per_layer.values()
        ]) if per_layer else np.array([], dtype=np.float32)
 
        layer_activations_frozen = tuple(
            (name, tuple(float(v) for v in vals))
            for name, vals in per_layer.items()
        )
 
        # Resolve predicted index to the same type as the dataset labels
        resolved_predicted = self._resolve_predicted(predicted)
 
        return InferenceRecord(
            id                = record_id,
            input             = input_row,
            label             = label,
            predicted         = resolved_predicted,
            correct           = resolved_predicted == label,
            activations       = tuple(flat.tolist()),
            layer_activations = layer_activations_frozen
        )