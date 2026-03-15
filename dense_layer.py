import numpy as np
from activations import ACTIVATIONS


class DenseLayer:
    """
    A single fully-connected layer.
    Z = X @ W + b
    A = activation(Z)
    """
    def __init__(self, input_size, output_size, activation="relu"):
        # He initialization for ReLU, Xavier for others
        if activation == "relu":
            scale = np.sqrt(2.0 / input_size)
        else:
            scale = np.sqrt(1.0 / input_size)
 
        self.W = np.random.randn(input_size, output_size) * scale
        self.b = np.zeros((1, output_size))
        self.activation = ACTIVATIONS[activation]()
 
        # Momentum / Adam state (populated by optimizer)
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)
        self.mW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
 
    def forward(self, X):
        self.X = X                              # cache input for backprop
        self.Z = X @ self.W + self.b            # linear transform
        self.A = self.activation.forward(self.Z)
        return self.A
 
    def backward(self, dA):
        """
        Given dA (gradient of loss w.r.t. this layer's output),
        compute gradients for W, b, and pass dX back to previous layer.
        """
        dZ = self.activation.backward(dA)       # through activation
        n = self.X.shape[0]
        self.dW = self.X.T @ dZ / n             # gradient for weights
        self.db = np.sum(dZ, axis=0, keepdims=True) / n  # gradient for biases
        dX = dZ @ self.W.T                      # gradient to pass back
        return dX