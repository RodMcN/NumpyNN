from .. import optimisation
from .. import utils
from .. import layers


def train(model, x, y, loss_fn=None, opt='sgd', epochs=1, shuffle=True, batch_size=None, opt_args=None, loss_fn_args=None, verbose=False):
    if type(opt == str):
        opt = opt.lower()
        opt = utils.get_class(opt, optimisation)

    if type(loss_fn == str):
        loss_fn = loss_fn.lower()
        loss_fn = utils.get_class(loss_fn, layers)

    opt_args = opt_args or {}
    loss_fn_args = loss_fn_args or {}
    opt = opt(**opt_args)
    loss_fn = loss_fn(**loss_fn_args)

    losses = []

    for e in range(epochs):
        if shuffle:
            x, y = utils.shuffle(x, y)

        preds = model(x)
        loss = loss_fn(preds, y)
        losses.append(loss)
        if verbose:
            print(loss)

        model.backward(loss_fn)
        opt.update(model)

    return losses
