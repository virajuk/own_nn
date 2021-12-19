import numpy as np
import sys

from src.nn import NN
from src.activation import Sigmoid, SomethingElse, Relu


# sigm = Sigmoid()
#
# fuck = sigm(x=3)
# print(fuck)

kwargs = {
    "lr": .23,
    "activation": Relu()
}
nn = NN()

input_input = np.array([[3.0], [4.5], [1.0]])

nn.set_input_layer(input_input)
# print(nn.input_matrix)

nn.add_hidden_layer(4)
nn.add_hidden_layer(2)
nn.add_hidden_layer(5)
nn.set_output_layer(4)

x = np.array([[-1, -8, -5], [-12, 2, 7]])
# print(nn.activation_function(x=x))

print(repr(nn))
