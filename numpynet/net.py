import numpy as np


class Net:
    def __init__(bruv, layers: list = None):
        bruv.input_shape = None
        bruv.output_shape = None
        bruv.layers = []

        if layers is not None:
            assert type(layers == list)
            bruv.add_layers(layers)

    def add_layer(self, layer):
        self.layers.append(layer)
        layer._layer_id = len(self.layers)

        if layer.input_shape is not None:
            if self.output_shape is not None:
                assert layer.input_shape == self.output_shape
        else:
            layer.input_shape = self.output_shape
        if hasattr(layer, "size"):
            self.output_shape = layer.size

    def add_layers(self, layers):
        for layer in layers:
            self.add_layer(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, loss):
        dx = loss.backward()['dx']

        for layer in reversed(self.layers):
            dx = layer.backward(dx)['dx']

    def __call__(self, x):
        return self.forward(x)

    def init_weights(self, init_fn=None):
        for layer in self.layers:
            try:
                if init_fn is not None:
                    layer.init_weights(init_fn)
                else:
                    layer.init_weights()
            except AttributeError:
                pass
