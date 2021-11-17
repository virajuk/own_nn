import math
import numpy as np


def sigmoid_function(x):

    """
    y = 1(1 + e^(-x))
    :param x:
    :return: y
    """
    y = 1/(1+math.exp(-x))
    return y


def matrix_multiplication():

    a = np.array([[2, 5, 8], [4, 8, 3]])
    b = np.array([[1, 6, 4], [0, 7, 9], [3, 6, 1]])

    c = np.array([[2, 5, 8], [4, 8, 3]])

    np.matmul(a, b, out=c)



if __name__ == '__main__':

    matrix_multiplication()
