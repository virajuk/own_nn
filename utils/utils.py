import numpy as np


def sigmoid_function(x):

    """
    y = 1(1 + e^(-x))
    :param x:
    :return: y
    """

    y = 1 / (1 + np.exp(-x))
    return y

