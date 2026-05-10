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

    def extract_model_data(self, model: Any) -> ModelMetadata:
        layers = []
        for layer in model.layers:
            w = layer.get_weights()
            layers.append(LayerInfo(
                name               = layer.name,
                type               = type(layer).__name__,
                trainable          = layer.trainable,
                activation         = layer.activation.__name__ if hasattr(layer, 'activation') else None,
                num_neurons        = layer.units if hasattr(layer, 'units') else None,
                input_shape        = tuple(layer.input_shape)  if hasattr(layer, 'input_shape')  else None,
                output_shape       = tuple(layer.output_shape) if hasattr(layer, 'output_shape') else None,
                weight_shape       = tuple(w[0].shape) if w else None,
                bias_shape         = tuple(w[1].shape) if len(w) > 1 else None,
                num_weights        = w[0].size if w else 0,
                num_biases         = w[1].size if len(w) > 1 else 0,
                relevant_inference = not isinstance(layer, (keras.layers.Dropout,)),
            ))

        return ModelMetadata(
            format               = 'keras',
            total_params         = model.count_params(),
            trainable_params     = int(sum(tf.size(w).numpy() for w in model.trainable_weights)),
            non_trainable_params = int(sum(tf.size(w).numpy() for w in model.non_trainable_weights)),
            num_layers           = len(model.layers),
            input_shape          = tuple(model.input_shape),
            output_shape         = tuple(model.output_shape),
            layers               = tuple(layers),
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