# 3 layer 2 node network

import numpy as np

from utils import utils as ut

input_input = np.array([[3.0], [1.0]])
weight_input_hidden = np.array([[8.0, 3.0], [6.0, 5.0]])
weight_hidden_output = np.array([[1.0, 0.7], [1.6, 4.3]])

LEARNING_RATE = 0.1

# print(weight_hidden_output)
# print(weight_hidden_output.shape)

input_hidden = np.matmul(weight_input_hidden, input_input)
# print(input_hidden)

output_hidden = ut.sigmoid_function(input_hidden)
# print(output_hidden)

input_output = np.matmul(weight_hidden_output, output_hidden)
# print(input_output)

output_output = ut.sigmoid_function(input_output)
# print(output_output)

label = np.array([[1.0], [0.0]])
error_output = (label - output_output)**2
# print(error_output)


#######################################################################################
error_hidden = np.matmul(weight_hidden_output.T, error_output)
# print(error_hidden)

delta_weight_hidden_output = np.matmul(LEARNING_RATE*(-2)*error_output*(ut.sigmoid_function(input_output))*(1-ut.sigmoid_function(input_output)), output_hidden.T)
print(delta_weight_hidden_output)

# print(weight_hidden_output)
new_weight_hidden_output = weight_hidden_output + delta_weight_hidden_output
print(new_weight_hidden_output)

error_input = np.matmul(weight_input_hidden.T, error_hidden)
# print(error_input)

delta_weight_input_hidden = np.matmul(LEARNING_RATE*(-2)*error_hidden*(ut.sigmoid_function(input_hidden))*(1-ut.sigmoid_function(input_hidden)), input_hidden.T)
print(delta_weight_input_hidden)

new_weight_input_hidden = weight_input_hidden + delta_weight_input_hidden
print(new_weight_input_hidden)