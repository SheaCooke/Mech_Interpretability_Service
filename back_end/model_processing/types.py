
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class LayerInfo:
    """
    Metadata for a single model layer. Displayed in UI
    """
    name:               str
    type:               str
    activation:         Optional[str]
    num_neurons:        Optional[int]
    relevant_inference: bool


@dataclass(frozen=True)
class ModelMetadata:
    """
    summary of a loaded models architecture and parameters
    """
    format:               str
    total_params:         int
    num_layers:           int
    input_shape:          Optional[tuple]
    output_shape:         Optional[tuple]
    layers:               tuple[LayerInfo, ...]

    def to_dict(self) -> dict:
        """Serialise to a plain dict suitable for JSON responses."""
        return {
            'format':               self.format,
            'total_params':         self.total_params,
            'num_layers':           self.num_layers,
            'input_shape':          list(self.input_shape) if self.input_shape else None,
            'output_shape':         list(self.output_shape) if self.output_shape else None,
            'layers': [
                {
                    'name':               l.name,
                    'type':               l.type,
                    'activation':         l.activation,
                    'num_neurons':        l.num_neurons,
                    'relevant_inference': l.relevant_inference,
                }
                for l in self.layers
            ],
        }


@dataclass(frozen=True)
class DataRecord: #TODO: store in file
    """A single input record loaded from a dataset file."""
    id:    str
    input: tuple 
    label: Optional[int|float|str]


@dataclass(frozen=True)
class InferenceRecord:
    """
    The result of running a single DataRecord through a model.
    """
    id:                str
    input:             tuple
    label:             Optional[int|float|str]
    predicted:         int
    correct:           bool
    activations:       tuple 
    layer_activations: tuple 

    def activations_array(self):
        return np.array(self.activations, dtype=np.float32)

    def layer_activations_dict(self) -> dict[str, list[float]]:
        return {name: list(values) for name, values in self.layer_activations}

    def to_dict(self) -> dict: 
        return {
            'id':                self.id,
            'input':             list(self.input),
            'label':             self.label,
            'predicted':         self.predicted,
            'correct':           self.correct,
            'activations':       self.activations_array(),
            'layer_activations': self.layer_activations_dict(),
        }