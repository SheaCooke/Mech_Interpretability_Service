import numpy as np
import pickle
from dense_layer import DenseLayer
from loss_functions import LOSSES
from optimizers import OPTIMIZERS


class NeuralNetwork:
    """
    Flexible feedforward neural network.
 
    Parameters
    ----------
    layer_configs : list of dicts
        Each dict defines a layer, e.g.:
        {"input_size": 2, "output_size": 16, "activation": "relu"}
    loss : str
        Loss function name: "mse", "binary_crossentropy", "categorical_crossentropy"
    optimizer : str
        Optimizer name: "sgd", "sgd_momentum", "adam"
    optimizer_params : dict
        Keyword arguments forwarded to the optimizer constructor.
    """
    def __init__(self, layer_configs, loss="mse", optimizer="adam", optimizer_params=None):
        self.layers = [
            DenseLayer(**cfg) for cfg in layer_configs
        ]
        self.loss_fn = LOSSES[loss]()
        opt_cls = OPTIMIZERS[optimizer]
        self.optimizer = opt_cls(**(optimizer_params or {}))
        self.history = {"loss": [], "accuracy": []}
        self.t = 0  # global step counter for Adam
 
    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, X):
        A = X 
        for layer in self.layers:
            A = layer.forward(A)
        return A
 
    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------
    def backward(self, Y_pred, Y_true):
        dA = self.loss_fn.backward()
        for layer in reversed(self.layers):
            dA = layer.backward(dA)
 
    # ------------------------------------------------------------------
    # Weight update
    # ------------------------------------------------------------------
    def update_weights(self):
        self.t += 1
        for layer in self.layers:
            self.optimizer.update(layer, t=self.t)
 
    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, X, Y, epochs=1000, batch_size=None, verbose=True, print_every=100):
        n = X.shape[0]
        batch_size = batch_size or n  # default: full-batch gradient descent
 
        for epoch in range(1, epochs + 1):
            # Shuffle data each epoch
            indices = np.random.permutation(n)
            X_shuffled, Y_shuffled = X[indices], Y[indices]
 
            epoch_loss = 0.0
            num_batches = 0
 
            for start in range(0, n, batch_size):
                X_batch = X_shuffled[start:start + batch_size]
                Y_batch = Y_shuffled[start:start + batch_size]
 
                Y_pred = self.forward(X_batch)
                batch_loss = self.loss_fn.forward(Y_pred, Y_batch)
                self.backward(Y_pred, Y_batch)
                self.update_weights()
 
                epoch_loss += batch_loss
                num_batches += 1
 
            avg_loss = epoch_loss / num_batches
            self.history["loss"].append(avg_loss)
 
            if verbose and epoch % print_every == 0:
                acc = self._accuracy(X, Y)
                self.history["accuracy"].append(acc)
                print(f"Epoch {epoch:>6} | Loss: {avg_loss:.6f} | Accuracy: {acc:.4f}")
 
    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, X):
        """Run forward pass and return raw output."""
        return self.forward(X)
 
    def predict_classes(self, X):
        """Return predicted class indices."""
        probs = self.predict(X)
        if probs.shape[1] == 1:
            return (probs >= 0.5).astype(int).flatten()
        return np.argmax(probs, axis=1)
 
    # ------------------------------------------------------------------
    # Accuracy helper
    # ------------------------------------------------------------------
    def _accuracy(self, X, Y):
        preds = self.predict_classes(X)
        if Y.ndim > 1 and Y.shape[1] > 1:
            true = np.argmax(Y, axis=1)
        else:
            true = Y.flatten().astype(int)
        return np.mean(preds == true)
 
    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save(self, filepath):
        """Serialize weights and biases to a pickle file."""
        state = {
            "weights": [layer.W for layer in self.layers],
            "biases":  [layer.b for layer in self.layers],
            "t": self.t,
            "history": self.history,
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f, protocol=5)
        print(f"Model saved to {filepath}")
 
    def load(self, filepath):
        """Load weights and biases from a pickle file."""
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        for i, layer in enumerate(self.layers):
            layer.W = state["weights"][i]
            layer.b = state["biases"][i]
        self.t = state["t"]
        self.history = state["history"]
        print(f"Model loaded from {filepath}")