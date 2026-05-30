
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import numpy as np

from .types import ModelMetadata, InferenceRecord, DataRecord


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

    def run_inference(
        self,
        model:    Any,
        records:  list[DataRecord],
    ) -> list[InferenceRecord]:

        self._build_class_map(records)
        self._prepare(model)
        results = []

        try:
            for record in records:
                tensor           = self._prepare_input(record.input)
                raw_out, per_layer = self._forward(model, tensor)
                predicted        = self._get_prediction(raw_out)
                inf_record       = self._build_record(record, predicted, per_layer)
                results.append(inf_record)
        finally:
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

    def _build_class_map(self, records: list) -> None:
        """
        Inspect the dataset labels to determine whether a class name mapping is needed
        """
        labels = [r.label for r in records if r.label is not None]
        if not labels:
            self._class_names = {}
            return
 
        if isinstance(labels[0], int):
            # Integer labels, predicted index IS the label, no mapping needed
            self._class_names = {}
            return
 
        # string or float labels, sort unique values to reconstruct index order
        unique = sorted(set(labels), key=lambda x: (str(type(x)), x))
        self._class_names = {idx: val for idx, val in enumerate(unique)}

    def _build_record(
        self,
        record: DataRecord,
        predicted: int,
        per_layer: dict[str, list[float]],
    ) -> InferenceRecord:

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
            id                = record.id,
            input             = record.input,
            label             = record.label,
            predicted         = resolved_predicted,
            correct           = resolved_predicted == record.label,
            activations       = tuple(flat.tolist()),
            layer_activations = layer_activations_frozen,
        )