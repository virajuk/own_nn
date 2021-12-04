import numpy as np

from utils.utils import sigmoid_function

input_matrix = np.array([[0.9], [0.1], [0.8]])

weight_input_hidden = np.array([[0.9, 0.3, 0.4], [0.2, 0.8, 0.2], [0.1, 0.5, 0.6]])

weight_hidden_output = np.array([[0.3, 0.7, 0.5], [0.6, 0.5, 0.2], [0.8, 0.1, 0.9]])

# print(input_matrix)
# print(weight_input_hidden)
# print(weight_hidden_output)

x_hidden = np.matmul(weight_input_hidden, input_matrix)
print(x_hidden)

o_hidden = sigmoid_function(x_hidden)
print(o_hidden)
