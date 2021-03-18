from torch import optim

def optimizerMaker(optimizer:str, learningRate:float, **kwargs):
    """
    Stores variables and returns a function type optim(params) to call
    each time a new optimizer is required for diferent set of parameters.
    """
    def adamMaker(params):
        return optim.Adam(params, lr=learningRate, **kwargs)
    def rmspropMaker(params):
        return optim.RMSprop(params, lr=learningRate, **kwargs)

    if optimizer == "adam":
        return adamMaker
    elif optimizer == "rmsprop":
        return rmspropMaker