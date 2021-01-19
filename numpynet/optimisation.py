import numpy as np


class SGD:

    shorthand = "sgd"

    def __init__(self, lr=0.01, momentum=False, nesterov=False):
        self.lr = lr

    def update(self, model=None, layers=None, gradients=None):
        assert (bool(model is None) ^ bool(layers is None)), "pass exactly one of model or layers"
        if model is not None:
            layers = model.layers

        if gradients is None:
            gradients = [l.gradients for l in layers]
        assert len(gradients) == len(layers)

        for layer in layers:
            grad_dict = layer.gradients
            if 'dw' in grad_dict:
                layer.weights -= self.lr * grad_dict['dw']
            if 'db' in grad_dict:
                layer.bias -= self.lr * grad_dict['db']


class Adam: # TODO ADAM
    pass