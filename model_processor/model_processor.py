import tensorflow as tf
import keras
from activations import ActivationFunction

class Model_Processor:
    def __init__(self, file_path='test/test_model.keras'): #TODO: update default path after initial testing
        self.model = self.__upload_model(file_path)
        self.weights: list[list[float]] = self.__extract_weights() #weights by layer
        self.biases: list[list[float]] = self.__extract_biases() #biases by layer
        self.activation_functions: list[ActivationFunction] = self.__extract_activation_functions() #activation functions by layer
        self.parameters: dict = self.__extract_parameters()
    
    def __upload_model(self, file_path: str):
        file_type = file_path.split('.')[-1]
        model = None
        if file_type == 'keras':
            model = keras.saving.load_model(file_path)
        #TODO: extend with more file types
        return model

#TODO: update other methods to work with more than just Keras
    def __extract_weights(self) -> list[list[float]]:
        weights = []
        for layer in self.model.layers:
            layer_weights = layer.get_weights()
            if layer_weights:  # some layers (e.g. dropout) have no weights
                weights.append(layer_weights[0].tolist())  # index 0 = weight matrix
        return weights

    def __extract_biases(self) -> list[list[float]]:
        biases = []
        for layer in self.model.layers:
            layer_weights = layer.get_weights()
            if len(layer_weights) > 1:  # index 1 = bias vector (if it exists)
                biases.append(layer_weights[1].tolist())
        return biases

    def __extract_activation_functions(self) -> list[ActivationFunction]:
        activation_functions = []
        for layer in self.model.layers:
            if hasattr(layer, 'activation'): #TODO: map to activation function class
                activation_name = layer.activation.__name__
                activation_functions.append(activation_name)
        return activation_functions

    def __extract_parameters(self) -> dict:
        return {
            'total_params': self.model.count_params(),
            'trainable_params': sum(tf.size(w).numpy() for w in self.model.trainable_weights),
            'non_trainable_params': sum(tf.size(w).numpy() for w in self.model.non_trainable_weights),
            'num_layers': len(self.model.layers),
            'layer_names': [layer.name for layer in self.model.layers],
            'layer_types': [type(layer).__name__ for layer in self.model.layers],
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
        }

  