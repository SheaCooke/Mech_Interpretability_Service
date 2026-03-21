from neural_network import NeuralNetwork
from typing import Any
import numpy as np

class Analyzer:
    def __init__(self, model: NeuralNetwork, test_data: list[Any]):
        self.model = model
        self.post_activations = self.get_post_activations(test_data)
        #TODO: replace with dictionary that maps something to identify the record (lebel?) to its vector
        self.post_activations_vector = self.create_vector_from_activations(self.post_activations)
    
    #TODO: update to handle test data > 1 record. should call create_vector_from_activations and store the values
    def get_post_activations(self, x) -> dict[str, list[list[float]]]:
        activations = {}
        self.model.forward(x)
        for i, layer in enumerate(self.model.layers):
            activations[f'layer_{i}_A'] = layer.A
        return activations

    
    def get_activations(self, x):
        activations = {}
        self.forward(x)
        for i, layer in enumerate(self.layers):
            activations[f"layer_{i}_Z"] = layer.Z   # pre-activation (linear output)
            activations[f"layer_{i}_A"] = layer.A   # post-activation
        return activations
    
    def create_vector_from_activations(self, activations: dict[str, list[list[float]]]) -> list[float]:
        return np.concatenate([v.flatten() for v in activations.values()])




