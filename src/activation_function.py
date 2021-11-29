import math
import numpy as np


class Activation:

    def sigmoid_function(self, x):
        """
            y = 1(1 + e^(-x))
            :param x:
            :return: y
            """
        y = 1 / (1 + math.exp(-x))
        return y

