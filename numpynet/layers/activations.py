from .layer import Activation
import numpy as np


class Sigmoid(Activation):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = 1 / (1 + np.exp(-x))
        self._cache = np.mean(x, axis=0, keepdims=True)
        return x

    def backward(self, prop_grad):
        dX = ((1 - self._cache) * self._cache)
        dX *= prop_grad

        gradients = {'dx': dX}

        if self.retain_gradients:
            self.gradients = gradients

        return gradients

    def update(self):
        pass


class Silu(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        sig = (1 + np.exp(-x))
        swish = x * sig
        self._cache = swish, sig
        return swish

    def backward(self, prop_grad):
        swish, sig = self._cache
        swish = np.mean(swish, axis=0, keepdims=True)
        sig = np.mean(sig, axis=0, keepdims=True)
        dx = swish + sig * (1 - swish)
        dx *= prop_grad
        return {"dx": dx}


class Relu(Activation):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        self._cache = np.maximum(0, x)
        return self._cache

    def backward(self, prop_grad):
        dX = np.mean(self._cache > 0, axis=0, keepdims=True) * prop_grad
        gradients = {'dx': dX}
        if self.retain_gradients:
            self.gradients = gradients
        return gradients


class LeakyRelu(Activation):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        self._cache = np.np.where(x > 0, x, self.alpha * x)
        return self._cache

    def backward(self, prop_grad):
        dX = np.mean(
            np.where(self._cache > 0, 1, self.alpha),
            axis=0, keepdims=True) * prop_grad
        gradients = {'dx': dX}
        if self.retain_gradients:
            self.gradients = gradients
        return gradients


class Softmax(Activation): #TODO
    def forward(self, x):
        # nprmalising by subtracting np.max(x)
        # helps prevent overflow
        exp = np.exp(x - np.max(x))
        return exp / np.sum(exp, axis=0, keepdims=True)

    def backward(self, x):
        pass

    def update(self):
        pass
