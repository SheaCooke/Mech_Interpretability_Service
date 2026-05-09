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
    def _get_prediction(self, raw_output: Any) -> int:
        """
        Extract the predicted class index from the raw model output.
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
        """
        flat = np.concatenate([
            np.array(v, dtype=np.float32).flatten()
            for v in per_layer.values()
        ]) if per_layer else np.array([], dtype=np.float32)

        # Store per-layer data as a tuple of (name, tuple) pairs so the
        # InferenceRecord remains frozen/hashable
        layer_activations_frozen = tuple(
            (name, tuple(float(v) for v in vals))
            for name, vals in per_layer.items()
        )

        return InferenceRecord(
            id                = record.id,
            input             = record.input,
            label             = record.label,
            predicted         = predicted,
            correct           = predicted == record.label,
            activations       = tuple(flat.tolist()),
            layer_activations = layer_activations_frozen,
        )