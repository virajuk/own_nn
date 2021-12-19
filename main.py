import numpy as np

from src.nn import NN

nn = NN(0.23)

input_input = np.array([[3.0], [4.5], [1.0]])

nn.input_layer(input_input)
# print(nn.input_matrix)

nn.add_hidden_layer(4)
nn.add_hidden_layer(2)
nn.add_hidden_layer(5)
nn.output_layer(2)

nn.summary()