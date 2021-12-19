import numpy as np


class Sigmoid:

    def __init__(self):
        pass

    def __call__(self, **kwargs):
        """
        y = 1(1 + e^(-x))
        :param x:
        :return: y
        """

        y = 1 / (1 + np.exp(-kwargs["x"]))
        return y
