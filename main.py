import numpy as np

from src.nn import NN
from src.activation import Sigmoid, SomethingElse


nn = NN(0.23)

input_input = np.array([[3.0], [4.5], [1.0]])

nn.set_input_layer(input_input)
# print(nn.input_matrix)

nn.add_hidden_layer(4)
nn.add_hidden_layer(2)
nn.add_hidden_layer(5)
nn.set_output_layer(4)

# nn.set_activation_function(SomethingElse)

nn.summary()
