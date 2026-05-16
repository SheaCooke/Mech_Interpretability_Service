"""
types.py

Immutable value objects used throughout the model processor pipeline.
Using frozen dataclasses guarantees thread safety — Python raises
FrozenInstanceError if anything attempts to mutate an instance after
creation, eliminating the entire class of race condition that arises
when multiple API requests share mutable state.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class LayerInfo:
    """Metadata for a single model layer."""
    name:               str
    type:               str
    trainable:          bool
    activation:         Optional[str]
    num_neurons:        Optional[int]
    input_shape:        Optional[tuple]
    output_shape:       Optional[tuple]
    weight_shape:       Optional[tuple]
    bias_shape:         Optional[tuple]
    num_weights:        int
    num_biases:         int
    relevant_inference: bool


@dataclass(frozen=True)
class ModelMetadata:
    """
    Immutable summary of a loaded model's architecture and parameters.
    Produced once at load time and shared safely across concurrent requests.
    """
    format:               str
    total_params:         int
    trainable_params:     int
    non_trainable_params: int
    num_layers:           int
    input_shape:          Optional[tuple]
    output_shape:         Optional[tuple]
    layers:               tuple[LayerInfo, ...]   # tuple — immutable, not list

    def to_dict(self) -> dict:
        """Serialise to a plain dict suitable for JSON responses."""
        return {
            'format':               self.format,
            'total_params':         self.total_params,
            'trainable_params':     self.trainable_params,
            'non_trainable_params': self.non_trainable_params,
            'num_layers':           self.num_layers,
            'input_shape':          list(self.input_shape) if self.input_shape else None,
            'output_shape':         list(self.output_shape) if self.output_shape else None,
            'layers': [
                {
                    'name':               l.name,
                    'type':               l.type,
                    'trainable':          l.trainable,
                    'activation':         l.activation,
                    'num_neurons':        l.num_neurons,
                    'input_shape':        list(l.input_shape) if l.input_shape else None,
                    'output_shape':       list(l.output_shape) if l.output_shape else None,
                    'weight_shape':       list(l.weight_shape) if l.weight_shape else None,
                    'bias_shape':         list(l.bias_shape) if l.bias_shape else None,
                    'num_weights':        l.num_weights,
                    'num_biases':         l.num_biases,
                    'relevant_inference': l.relevant_inference,
                }
                for l in self.layers
            ],
        }


@dataclass(frozen=True)
class DataRecord:
    """A single input record loaded from a dataset file."""
    id:    str
    input: tuple          # immutable; converted from np.ndarray at load time
    label: Optional[int|float|str]


@dataclass(frozen=True)
class InferenceRecord: #TODO: use this instead of a new data structure
    """
    The result of running a single DataRecord through a model.
    Immutable — safe to share across threads and pass between pipeline stages
    without defensive copying.
    """
    id:                str
    input:             tuple
    label:             Optional[int|float|str]
    predicted:         int
    correct:           bool
    activations:       tuple           # flat concatenated vector — for distance matrix
    layer_activations: tuple           # ((layer_name, (float, ...)), ...) — for prototype analysis

    def activations_array(self):
        """Return activations as a numpy array (lazy, not stored)."""
        import numpy as np
        return np.array(self.activations, dtype=np.float32)

    def layer_activations_dict(self) -> dict[str, list[float]]:
        """Return per-layer activations as a plain dict."""
        return {name: list(values) for name, values in self.layer_activations}

    def to_dict(self) -> dict: 
        """Serialise to a plain dict for JSON responses and downstream analysis."""
        return {
            'id':                self.id,
            'input':             list(self.input),
            'label':             self.label,
            'predicted':         self.predicted,
            'correct':           self.correct,
            'activations':       self.activations_array(),
            'layer_activations': self.layer_activations_dict(),
        }