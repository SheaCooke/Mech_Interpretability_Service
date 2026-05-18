"""
base.py

Abstract base class that defines:

  1. The Strategy interface — every concrete strategy (Keras, ONNX, PyTorch)
     must implement load(), extract_model_data(), and the four abstract steps
     of the inference pipeline.

  2. The Template Method for inference — run_inference() defines the fixed
     sequence (prepare → loop → forward → build record → teardown) and calls
     abstract methods for the steps that vary by format. This means the
     pipeline structure lives in exactly one place and subclasses only override
     what is different for their format.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import numpy as np

from .types import ModelMetadata, InferenceRecord, DataRecord


class ModelStrategy(ABC):
    """
    Strategy interface + Template Method base for all model formats.

    Subclasses implement the four abstract inference steps and the two
    abstract lifecycle methods (load, extract_model_data). The shared
    pipeline sequence in run_inference() is never duplicated.
    """

    def __init__(self):
        self._class_names: Optional[dict[int, Any]] = None

    @abstractmethod
    def load(self, file_path: str) -> Any:
        """
        Load a model from disk and return the framework-native model object.
        Should raise RuntimeError with a clear message on failure.
        """

    @abstractmethod
    def extract_model_data(self, model: Any) -> ModelMetadata:
        """
        Inspect a loaded model and return an immutable ModelMetadata snapshot.
        Called once at load time; result is shared safely across requests.
        """

    def run_inference(
        self,
        model:    Any,
        records:  list[DataRecord],
    ) -> list[InferenceRecord]:
        """
        Fixed inference pipeline sequence. Subclasses customise behaviour
        by overriding the four abstract steps below, not this method.

        Steps:
            1. _prepare(model)              — one-time setup before the loop
            2. for each record:
               a. _prepare_input(raw)       — reshape/cast to model's expected format
               b. _forward(model, tensor)   — forward pass, returns (output, per_layer)
               c. _get_prediction(output)   — extract predicted class index
               d. _build_record(...)        — assemble InferenceRecord
            3. _teardown(model)             — one-time cleanup after the loop
        """
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
        Must handle any necessary shape transformations (e.g. adding batch
        or channel dimensions).
        """

    @abstractmethod
    def _forward(self, model: Any, tensor: Any) -> tuple[Any, dict[str, list[float]]]:
        """
        Run a single forward pass.

        Returns:
            raw_out   — framework-native output (logits / probabilities)
            per_layer — {layer_name: [activation_float, ...]} for every layer
        """

    @abstractmethod
    def _prepare(self, model: Any) -> None:
        """
        One-time setup before the inference loop begins.
        Default: no-op. Override in strategies that need pre-loop initialisation
        (e.g. building a Keras activation model, registering PyTorch hooks).
        """

    @abstractmethod
    def _teardown(self, model: Any) -> None:
        """
        One-time cleanup after the inference loop completes.
        Default: no-op. Override in strategies that allocate resources in
        _prepare (e.g. removing PyTorch forward hooks).
        """

    def _get_prediction(self, raw_output: np.ndarray) -> int:
        """
        Extract the predicted class index from the raw model output.
        """
        return int(np.argmax(raw_output))
        
    def _resolve_predicted(self, predicted_idx: int) -> Any:
        """
        Map a predicted class index back to the original label type.
        For integer datasets returns the index unchanged.
        For string/float datasets returns the class name/value at that index.
        """
        if not self._class_names:
            return predicted_idx
        return self._class_names.get(predicted_idx, predicted_idx)

    def _build_class_map(self, records: list) -> None:
        """
        Inspect the dataset labels to determine whether a class name mapping
        is needed. Called lazily on the first record of each inference run.
 
        For integer-labelled datasets: store an empty dict (no mapping needed).
        For string/float-labelled datasets: build {int_index: label_value}
        by sorting the unique label values — this mirrors how the training
        script assigns indices (alphabetical for strings, ascending for floats).
        """
        labels = [r.label for r in records if r.label is not None]
        if not labels:
            self._class_names = {}
            return
 
        sample = labels[0]
        if isinstance(sample, int):
            # Integer labels, predicted index IS the label, no mapping needed
            self._class_names = {}
            return
 
        # String or float labels, sort unique values to reconstruct index order.
        # This matches sklearn's LabelEncoder and alphabetical class ordering.
        unique = sorted(set(labels), key=lambda x: (str(type(x)), x))
        self._class_names = {idx: val for idx, val in enumerate(unique)}

    def _build_record(
        self,
        record:    DataRecord,
        predicted: int,
        per_layer: dict[str, list[float]],
    ) -> InferenceRecord:
        """
        Assemble an immutable InferenceRecord from a forward pass result.
        Shared across all strategies — lives here once, not in each subclass.
 
        The flat activations vector is built by concatenating all per-layer
        outputs, preserving forward-pass layer order. This vector is used
        by Vector_Analyzer for cosine distance and cluster analysis.
 
        For datasets with string or float labels, _resolve_predicted() maps
        the integer argmax index back to the original label type so that:
          - correct = (predicted_label == true_label) works across types
          - the layer-wise analysis displays meaningful label names
        """
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