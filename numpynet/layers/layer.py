from abc import ABC, abstractmethod

class Layer(ABC):
    def __init__(self):
        self.input_shape = None
        self._cache = None
        self._layer_id = None
        self.calc_grad = True
        self.freeze_weights = False
        self.gradients = {}
        self.retain_gradients = True

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, x):
        pass

    def update(self):
        raise NotImplementedError

    def __call__(self, *args):
        return self.forward(*args)

    def __repr__(self):
        r = str(type(self))
        if self._layer_id is not None:
            r = f"{r} at layer: {str(self._layer_id)}"
        return r


class Activation(Layer, ABC):
    """
    Functionality is identical to a layer (for now) but provides distinction between Activations and
    other layers
    """
    def __init__(self):
        super().__init__()
        self.retain_gradients = False


class Loss(Layer, ABC):

    def __init__(self):
        super().__init__()
        self.retain_gradients = False

    def forward(self, x, target):
        pass
