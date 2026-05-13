"""
strategies/keras_strategy.py

Concrete Strategy for Keras (.keras) models.

Implements the four abstract inference steps defined in ModelStrategy:
  - _prepare:       builds the Keras activation model once before the loop
  - _prepare_input: adds the batch dimension
  - _forward:       runs predict() and returns (output, per_layer dict)
  - _get_prediction: argmax over the final layer output
  - _teardown:      no-op for Keras (no resources to release)

Also implements load() and extract_model_data().
"""

from __future__ import annotations
from typing import Any
import numpy as np
import keras
import tensorflow as tf

from ..base import ModelStrategy
from ..types import ModelMetadata, LayerInfo


class KerasStrategy(ModelStrategy):

    def __init__(self):
        # Activation model is built once in _prepare and held for the
        # duration of a single run_inference call. It is not stored as
        # instance state between calls so concurrent requests each get
        # their own activation model rather than sharing one.
        self._activation_model = None
        self._layer_names: list[str] = []
        self._class_names: Optional[dict[int, Any]] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self, file_path: str) -> Any:
        """
        Load a .keras model file, patching layer __init__ methods to
        tolerate unknown kwargs introduced by newer Keras versions
        (e.g. quantization_config from Keras 3.1+). The patch is applied
        and removed atomically so it cannot leak into other threads.
        """
        def make_patched_init(original):
            def patched_init(self, *args, **kwargs):
                known = original.__code__.co_varnames[:original.__code__.co_argcount]
                for k in [k for k in kwargs if k not in known]:
                    kwargs.pop(k)
                original(self, *args, **kwargs)
            return patched_init

        layers_to_patch = [
            keras.layers.Dense,
            keras.layers.Conv2D,
            keras.layers.LSTM,
            keras.layers.GRU,
            keras.layers.Embedding,
            keras.layers.BatchNormalization,
        ]
        originals = {layer: layer.__init__ for layer in layers_to_patch}
        for layer in layers_to_patch:
            layer.__init__ = make_patched_init(originals[layer])

        try:
            model = keras.saving.load_model(file_path)
        except TypeError as e:
            raise RuntimeError(f"Failed to load Keras model: {e}")
        finally:
            for layer, original in originals.items():
                layer.__init__ = original

        return model

    # Layer types whose behaviour differs at inference vs training time.
    # These are excluded from relevant_inference so the analysis tool
    # can correctly identify which layers participate in a forward pass.
    _TRAINING_ONLY_LAYERS = (
        keras.layers.Dropout,
        keras.layers.AlphaDropout,
        keras.layers.GaussianDropout,
        keras.layers.GaussianNoise,
        keras.layers.RandomTranslation,
        keras.layers.RandomRotation,
        keras.layers.RandomFlip,
        keras.layers.RandomZoom,
        keras.layers.RandomCrop,
    )
 
    # Layer types where get_weights() follows the [weight_matrix, bias] convention.
    # For all other types (BatchNorm, LayerNorm, Embedding etc.) the weight list
    # has a different structure and should not be reported as weight/bias shapes.
    _WEIGHT_BIAS_LAYERS = (
        keras.layers.Dense,
        keras.layers.Conv1D,
        keras.layers.Conv2D,
        keras.layers.Conv3D,
        keras.layers.LSTM,
        keras.layers.GRU,
        keras.layers.Embedding,
    )
    

    @staticmethod
    def _safe_model_shape(shape_attr: Any) -> Optional[tuple]:
        """
        Same as _safe_shape but applied to model-level input/output shapes.
        Multi-input models return a list of shapes; wrap each in a tuple.
        """
        try:
            if isinstance(shape_attr, list):
                return tuple(tuple(s) for s in shape_attr)
            return tuple(shape_attr)
        except Exception:
            return None

    @staticmethod
    def _safe_shape(shape_attr: Any) -> Optional[tuple]:
        """
        Safely convert a layer shape attribute to a tuple.
        Handles:
          - Single shapes: (None, 28, 28) → (None, 28, 28)
          - Multi-input/output layers that return a list of shapes:
            [(None, 32), (None, 32)] → ((None, 32), (None, 32))
          - Unbuilt layers or any other exception → None
        """
        try:
            if isinstance(shape_attr, list):
                # Multi-input or multi-output layer
                return tuple(tuple(s) for s in shape_attr)
            return tuple(shape_attr)
        except Exception:
            return None
            
    @staticmethod
    def _safe_activation(layer: Any) -> Optional[str]:
        """
        Extract the activation function name from a layer.
        Handles layers where activation is a callable, a string, or absent.
        Returns None rather than raising for any unexpected type.
        """
        if not hasattr(layer, 'activation'):
            return None
        act = layer.activation
        # Most layers store activation as a callable with __name__
        if callable(act) and hasattr(act, '__name__'):
            return act.__name__
        # Some versions store it as a string directly
        if isinstance(act, str):
            return act
        # Fallback: convert to string representation
        try:
            return str(act)
        except Exception:
            return None

    def extract_model_data(self, model: Any) -> ModelMetadata:
        layer_infos = []
 
        for layer in model.layers:
            w = layer.get_weights()
 
            # Weight and bias shapes are only meaningful for layer types that
            # follow the [kernel, bias] convention. For BatchNorm, LayerNorm,
            # and similar layers, report total param count only.
            weight_shape = None
            bias_shape   = None
            num_weights  = 0
            num_biases   = 0
 
            if w:
                if isinstance(layer, self._WEIGHT_BIAS_LAYERS):
                    weight_shape = tuple(w[0].shape)
                    num_weights  = w[0].size
                    has_bias = getattr(layer, 'use_bias', False)
                    if has_bias and len(w) > 1:
                        bias_shape = tuple(w[1].shape)
                        num_biases = w[1].size
                else:
                    # BatchNorm, LayerNorm, Embedding etc. — count all params
                    num_weights = sum(arr.size for arr in w)
 
            layer_infos.append(LayerInfo(
                name               = layer.name,
                type               = type(layer).__name__,
                trainable          = layer.trainable,
                activation         = self._safe_activation(layer),
                num_neurons        = getattr(layer, 'units', None),
                input_shape        = self._safe_shape(layer.input_shape)
                                     if hasattr(layer, 'input_shape') else None,
                output_shape       = self._safe_shape(layer.output_shape)
                                     if hasattr(layer, 'output_shape') else None,
                weight_shape       = weight_shape,
                bias_shape         = bias_shape,
                num_weights        = num_weights,
                num_biases         = num_biases,
                relevant_inference = not isinstance(layer, self._TRAINING_ONLY_LAYERS),
            ))
 
        return ModelMetadata(
            format               = 'keras',
            total_params         = model.count_params(),
            trainable_params     = int(sum(tf.size(w).numpy() for w in model.trainable_weights)),
            non_trainable_params = int(sum(tf.size(w).numpy() for w in model.non_trainable_weights)),
            num_layers           = len(model.layers),
            input_shape          = self._safe_model_shape(model.input_shape),
            output_shape         = self._safe_model_shape(model.output_shape),
            layers               = tuple(layer_infos),
        )


    def _prepare(self, model: Any) -> None:
        """
        Build the activation model once before the inference loop.
        Keras allows outputting every layer's tensor in a single predict()
        call by constructing a Model with multiple outputs.
        """
        self._layer_names = [layer.name for layer in model.layers]
        self._activation_model = keras.Model(
            inputs=model.inputs,
            outputs=[layer.output for layer in model.layers],
        )

    def run_inference(self, model: Any, records: list) -> list:
        """
        Override to build the class name map before the inference loop begins.
        For integer-labelled datasets this is a no-op. For string/float-labelled
        datasets it reconstructs the index→label mapping so _resolve_predicted
        can convert argmax integers back to meaningful label values.
        """
        self._build_class_map(records)
        return super().run_inference(model, records)

    def _prepare_input(self, raw: tuple) -> np.ndarray:
        """Add batch dimension. Keras Flatten handles any 2D/3D input shape."""
        return np.expand_dims(np.array(raw, dtype=np.float32), axis=0)

    def _forward(
        self,
        model: Any,
        tensor: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, list[float]]]:
        """
        Run a single forward pass through the activation model.
        Returns (final_layer_output, per_layer_dict).
        """
        layer_outputs = self._activation_model.predict(tensor, verbose=0)

        per_layer = {
            name: output[0].flatten().tolist()
            for name, output in zip(self._layer_names, layer_outputs)
        }

        return layer_outputs[-1][0], per_layer

    def _get_prediction(self, raw_output: np.ndarray) -> int:
        return int(np.argmax(raw_output))

    def _teardown(self, model: Any) -> None:
        """Release the activation model reference after the inference loop."""
        self._activation_model = None
        self._layer_names = []
        self._class_names = None

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
            # Integer labels — predicted index IS the label, no mapping needed
            self._class_names = {}
            return
 
        # String or float labels — sort unique values to reconstruct index order.
        # This matches sklearn's LabelEncoder and alphabetical class ordering.
        unique = sorted(set(labels), key=lambda x: (str(type(x)), x))
        self._class_names = {idx: val for idx, val in enumerate(unique)}

    def _resolve_predicted(self, predicted_idx: int) -> Any:
        """
        Map a predicted class index back to the original label type.
        For integer datasets returns the index unchanged.
        For string/float datasets returns the class name/value at that index.
        """
        if not self._class_names:
            return predicted_idx
        return self._class_names.get(predicted_idx, predicted_idx)