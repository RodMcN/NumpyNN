from .layer import Loss
import numpy as np


class MSELoss(Loss):

    shorthand = "mse"

    def __init__(self):
        super().__init__()

    def forward(self, x, target):
        self._cache = x - target
        return np.mean(self._cache ** 2, axis=0)

    def backward(self):
        dx = 2 / len(self._cache) * self._cache
        dx = np.mean(dx, axis=0, keepdims=True)
        gradients = {"dx":dx}
        if self.retain_gradients:
            self.gradients = gradients
        return gradients


class BCELoss(Loss):

    shorthand = "bce"

    def __init__(self):
        super().__init__()

    def forward(self, x, target):
        self._cache = x, target
        return -np.mean(np.nan_to_num((target * np.log(x)) + ((1 - target) * np.log(1-x))), axis=0)

    def backward(self):
        x, y =self._cache
        dx = ((1 - y) / (1-x)) - (y / x)
        dx = np.mean(dx, axis=0, keepdims=True)
        gradients = {"dx": dx}
        if self.retain_gradients:
            self.gradients = gradients
        return gradients

    def update(self):
        pass
