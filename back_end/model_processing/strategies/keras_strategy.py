"""
strategies/keras_strategy.py

Concrete Strategy for Keras (.keras) models.

Implements the four abstract inference steps defined in ModelStrategy:
  - _prepare:       builds the Keras activation model once before the loop
  - _prepare_input: adds the batch dimension
  - _forward:       runs predict() and returns (output, per_layer dict)
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

    #TODO: replace with dynamic process that is not hard coded
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

    def __init__(self):
        self._activation_model = None #same weights as the uploaded model, just exposes internal tensors
        self._layer_names: list[str] = []
        super().__init__()


    def load(self, file_path: str) -> Any:
        try:
            return keras.models.load_model(file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load Keras model: {e}")
    

    @staticmethod
    def _safe_model_shape(shape_attr: Any) -> Optional[tuple]:
        try:
            if isinstance(shape_attr, list):
                return tuple(tuple(s) for s in shape_attr)
            return tuple(shape_attr)
        except Exception:
            return None


    def extract_model_data(self, model: Any) -> ModelMetadata:
        layer_infos = []
 
        for layer in model.layers:

            activation = getattr(layer, "activation", None)

            layer_infos.append(LayerInfo(
                name               = layer.name,
                type               = type(layer).__name__,
                activation         = activation.__name__ if activation else "N/A",
                num_neurons        = getattr(layer, 'units', None),
                relevant_inference = not isinstance(layer, self._TRAINING_ONLY_LAYERS)
            ))
 
        return ModelMetadata(
            format               = 'keras',
            total_params         = model.count_params(),
            num_layers           = len(model.layers),
            input_shape          = self._safe_model_shape(model.input_shape),
            output_shape         = self._safe_model_shape(model.output_shape),
            layers               = tuple(layer_infos)
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
            outputs=[layer.output for layer in model.layers]
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
        layer_outputs = self._activation_model.predict(tensor, verbose=0) #TODO: should take workers=2 or more, and use_multiprocessing=True from configs

        per_layer = {
            name: output[0].flatten().tolist()
            for name, output in zip(self._layer_names, layer_outputs)
        }

        return layer_outputs[-1][0], per_layer


    def _teardown(self, model: Any) -> None:
        self._activation_model = None
        self._layer_names = []
        self._class_names = None