from .layer import Layer
import numpy as np
from numpynet import initialisers
from scipy.signal import convolve


class Dense(Layer):
    def __init__(self, size, input_shape=None):
        super().__init__()

        self.size = size
        self.input_shape = input_shape
        self.inputs = None

        self.weights = None
        self.bias = np.zeros(self.size)

    def init_weights(self, init_fn='xavier'):
        if type(init_fn == str):
            init_fn = initialisers.init_fns[init_fn]
        init_fn(self)

    def forward(self, x):

        if self.weights is None:
            self.init_weights()

        #self._cache = x
        self._cache = np.mean(x, axis=0, keepdims=True)

        # TODO vectorise over batches
        batch_size = x.shape[0]
        batch_out = np.empty((batch_size, self.size))
        for i in range(batch_size):
            batch_out[i] = np.matmul(self.weights.T, x[i]) + self.bias
        return batch_out


    def backward(self, prop_grad):

        #dw = np.matmul(grad, np.mean(self.cache, axis=0, keepdims=True))
        dW = np.matmul(self._cache.T, prop_grad) # for updating weights
        dX = np.matmul(prop_grad, self.weights.T) # for backprop
        dB = prop_grad.squeeze() # for updating bias

        gradients = {"dx": dX, "dw": dW, "db": dB}

        if self.retain_gradients:
            self.gradients = {"dx": dX, "dw": dW, "db": dB}

        return gradients

    def update(self):
        pass


class Conv1D(Layer):
    def __init__(self, kernel_size, filters, padding=None, input_shape=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.fiters = filters
        self.padding = padding

        self.size = None
        self.input_shape = input_shape
        self.inputs = None

        self.weights = None
        self.bias = np.zeros(self.size)

    def forward(self, x):
        pass

class Conv2D(Layer):
    def __init__(self):
        super().__init__()
        raise NotImplemented