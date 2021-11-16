import math


def sigmoid_function(x):

    """
    y = 1(1 + e^(-x))
    :param x:
    :return: y
    """
    y = 1/(1+math.exp(-x))
    return y


value = sigmoid_function(0.6)
print(255*255*255*255)

