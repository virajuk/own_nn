import numpy as np


class Sigmoid:

    def __init__(self):
        pass

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

    def __call__(self, **kwargs):
        """
        y = 1(1 + e^(-x))
        :param x:
        :return: y
        """

        y = 1 / (1 + np.exp(-kwargs["x"]))
        return y


class SomethingElse:

    def __init__(self):
        pass

    def __repr__(self):
        repr_str = self.__class__.__name__
        # repr_str += f'This is something else activation,'
        return repr_str

    def __call__(self, **kwargs):
        """
        y = 1(1 + e^(-x))
        :param x:
        :return: y
        """

        return np.exp(-kwargs["x"])
