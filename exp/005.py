import numpy as np

from utils import utils as ut

input_matrix = np.array([[0.9], [0.1], [0.8]])

weight_input_hidden = np.array([[0.9, 0.3, 0.4], [0.2, 0.8, 0.2], [0.1, 0.5, 0.6]])

weight_hidden_output = np.array([[0.3, 0.7, 0.5], [0.6, 0.5, 0.2], [0.8, 0.1, 0.9]])

x_hidden = np.matmul(weight_input_hidden, input_matrix)
# print(x_hidden)

o_hidden = ut.sigmoid_function(x_hidden)
# print(o_hidden)

x_output = np.matmul(weight_hidden_output, o_hidden)
# print(x_output)

output_matrix = ut.sigmoid_function(x_output)
print(output_matrix)

error_matrix = np.array([[1.5], [0.5], [0.3]])

