import numpy as np
from contextlib import contextmanager
import inspect

@contextmanager
def no_grad(model): # TODO
    try:
        yield resource
    finally:
        release_resource(resource)


class Nograd():
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        print('enter method called')
        # set all to no grad required
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # set all to grad required
        print('exit method called')

### can this be a function in the mode instead, eg with model.no_grad()


def shuffle(x, y=None):
    idx = np.arange(len(x))
    x = x[idx, ...]
    if y is not None:
        y = y[idx, ...]
        return x, y
    return x

def get_class(shorthand, module):
    for _, m in inspect.getmembers(module):
        try:
            if m.shorthand == shorthand:
                return m
        except AttributeError:
            pass