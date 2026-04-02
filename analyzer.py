from neural_network import NeuralNetwork
from typing import Any, Iterable
import numpy as np
from scipy.spatial.distance import cdist

class Analyzer:
    def __init__(self, model: NeuralNetwork, test_data: list[Any]):
        self.model = model
        self.activation_vectors = self.get_post_activations_for_records(test_data) # map the activation vector to the record that produced it
        self.cosine_distance = self.find_cosine_distance_between_vectors(self.activation_vectors)
    

    def get_post_activations_for_records(self, x: Iterable[np.ndarray]) -> dict[tuple, np.ndarray]:
        post_act_records = {}
        for record in x: 
            self.model.forward(record)
            activation_values = []

            for layer in self.model.layers:
                activation_values.append(layer.A)

            flattened_vector = np.concatenate(activation_values, axis=None)
            vector = tuple(flattened_vector)
            post_act_records[vector] = record

        return post_act_records


    """
    Forward pass test data through all the layers, then go through and collect the cached activation values.
    """
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
    

    def find_cosine_distance_between_vectors(self, vectors: dict[tuple, np.ndarray]) -> np.ndarray:
        organized_vectors = np.vstack(vectors) #iterates over the keys to create arrays

        #distance_matrix[0][1] = cosine distance between organized_vectors[0] and organized_vectors[1]
        # 0 = identical, 1.0 = orthogonal, 2.0 = opposite
        distance_matrix = cdist(organized_vectors, organized_vectors, metric='cosine')
        #TODO: how do we associate the cosine distance back to the original vectors and records?
        #TODO: experiment with dot product instead of cosine distance. make option configurable

        return distance_matrix

    #TODO: algorithm to sort cosine distances, display the most similar records based on the activation
    #TODO: also display results with a clustering algorithm







