import numpy as np
from abc import ABC, abstractmethod
 
 
class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, data):
        pass

    @abstractmethod
    def backward(self, data):
        pass


class ReLU(ActivationFunction):
    def forward(self, Z):
        self.cache = Z
        return np.maximum(0, Z)
 
    def backward(self, dA):
        return dA * (self.cache > 0)
 
 
class Sigmoid(ActivationFunction):
    def forward(self, Z):
        self.cache = 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
        return self.cache
 
    def backward(self, dA):
        s = self.cache
        return dA * s * (1 - s)
 
 
class Tanh(ActivationFunction):
    def forward(self, Z):
        self.cache = np.tanh(Z)
        return self.cache
 
    def backward(self, dA):
        return dA * (1 - self.cache ** 2)
 
 
class Softmax(ActivationFunction):
    def forward(self, Z):
        # Subtract max for numerical stability
        shifted = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(shifted)
        self.cache = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        return self.cache
 
    def backward(self, dA):
        # When paired with cross-entropy loss, the gradient simplifies.
        # Full Jacobian version for standalone use:
        batch_size = dA.shape[0]
        dZ = np.zeros_like(dA)
        for i in range(batch_size):
            s = self.cache[i].reshape(-1, 1)           # (C, 1)
            jacobian = np.diagflat(s) - s @ s.T         # (C, C)
            dZ[i] = jacobian @ dA[i]
        return dZ
 
 
ACTIVATIONS = {
    "relu": ReLU,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "softmax": Softmax,
}
 