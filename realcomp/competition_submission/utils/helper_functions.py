import numpy as np


def mse(y, yh):
    return np.square(y - yh).mean()
