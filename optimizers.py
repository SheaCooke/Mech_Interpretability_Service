import numpy as np


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
 
    def update(self, layer, t=None):
        layer.W -= self.lr * layer.dW
        layer.b -= self.lr * layer.db
 
 
class SGDMomentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
 
    def update(self, layer, t=None):
        layer.vW = self.momentum * layer.vW - self.lr * layer.dW
        layer.vb = self.momentum * layer.vb - self.lr * layer.db
        layer.W += layer.vW
        layer.b += layer.vb
 
 
class Adam:
    """
    Adaptive Moment Estimation optimizer.
    Maintains per-parameter first (mean) and second (variance) moment estimates.
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
 
    def update(self, layer, t):
        b1, b2, eps = self.beta1, self.beta2, self.epsilon
 
        # Update biased first moment estimate
        layer.mW = b1 * layer.mW + (1 - b1) * layer.dW
        layer.mb = b1 * layer.mb + (1 - b1) * layer.db
 
        # Update biased second raw moment estimate
        layer.vW = b2 * layer.vW + (1 - b2) * layer.dW ** 2
        layer.vb = b2 * layer.vb + (1 - b2) * layer.db ** 2
 
        # Bias-corrected estimates
        mW_hat = layer.mW / (1 - b1 ** t)
        mb_hat = layer.mb / (1 - b1 ** t)
        vW_hat = layer.vW / (1 - b2 ** t)
        vb_hat = layer.vb / (1 - b2 ** t)
 
        # Parameter update
        layer.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + eps)
        layer.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + eps)
 
 
OPTIMIZERS = {
    "sgd": SGD,
    "sgd_momentum": SGDMomentum,
    "adam": Adam,
}