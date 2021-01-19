from .layer import Layer
import numpy as np


class Dropout(Layer):
    def __init__(self, p=0.5):
        super().__init__()
        self._p = p

    def forward(self, x):
        self._cache = np.where(
            np.random.uniform(0, 1, x.shape) > self._p,
            x, 0)
        return self._cache

    def backward(self, x):
        return {"dx": np.where(self._cache > 0, 1, 0)}

    def update(self):
        pass


class BatchNorm1D(Layer):
    pass
    #TODO