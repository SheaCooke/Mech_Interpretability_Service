import numpy as np


class MSELoss:
    """Mean Squared Error — for regression."""
    def forward(self, Y_pred, Y_true):
        self.Y_pred = Y_pred
        self.Y_true = Y_true
        return np.mean((Y_pred - Y_true) ** 2)
 
    def backward(self):
        n = self.Y_pred.shape[0]
        return 2 * (self.Y_pred - self.Y_true) / n
 
 
class BinaryCrossEntropyLoss:
    """Binary Cross-Entropy — for binary classification with sigmoid output."""
    def forward(self, Y_pred, Y_true):
        self.Y_pred = np.clip(Y_pred, 1e-9, 1 - 1e-9)
        self.Y_true = Y_true
        return -np.mean(
            Y_true * np.log(self.Y_pred) + (1 - Y_true) * np.log(1 - self.Y_pred)
        )
 
    def backward(self):
        n = self.Y_pred.shape[0]
        return (self.Y_pred - self.Y_true) / (self.Y_pred * (1 - self.Y_pred) * n)
 
 
class CategoricalCrossEntropyLoss:
    """
    Categorical Cross-Entropy — for multi-class classification with softmax output.
    Expects Y_true as one-hot encoded matrix OR integer class labels.
    """
    def forward(self, Y_pred, Y_true):
        self.Y_pred = np.clip(Y_pred, 1e-9, 1 - 1e-9)
        n = Y_pred.shape[0]
        # Convert integer labels to one-hot
        if Y_true.ndim == 1:
            one_hot = np.zeros_like(Y_pred)
            one_hot[np.arange(n), Y_true.astype(int)] = 1
            self.Y_true = one_hot
        else:
            self.Y_true = Y_true
        return -np.sum(self.Y_true * np.log(self.Y_pred)) / n
 
    def backward(self):
        n = self.Y_pred.shape[0]
        # Combined softmax + cross-entropy gradient: (Y_pred - Y_true) / n
        return (self.Y_pred - self.Y_true) / n
 
 
LOSSES = {
    "mse": MSELoss,
    "binary_crossentropy": BinaryCrossEntropyLoss,
    "categorical_crossentropy": CategoricalCrossEntropyLoss,
}