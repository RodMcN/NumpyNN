import numpy as np


def normal(layer, mean=0, std=0.1):
    layer.weights = np.random.normal(mean, std, (layer.input_shape, layer.size))
    layer.bias = np.ones(layer.size)

def xavier(layer):
    xavier_normal(layer)


def xavier_normal(layer):
    std = np.sqrt(2 / (layer.input_shape + layer.size))
    layer.weights = np.random.normal(0, std, (layer.input_shape, layer.size))


init_fns = {"normal": normal, "xavier": xavier, }