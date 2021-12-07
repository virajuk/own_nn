import numpy as np

from utils import utils as ut

input_matrix = np.array([[0.9], [0.1], [0.8]])

weight_input_hidden = np.array([[0.9, 0.3, 0.4], [0.2, 0.8, 0.2], [0.1, 0.5, 0.6]])
weight_hidden_output = np.array([[0.3, 0.7, 0.5], [0.6, 0.5, 0.2], [0.8, 0.1, 0.9]])
# print(weight_hidden_output.shape)

x_hidden = np.matmul(weight_input_hidden, input_matrix)
# print(x_hidden)

# print(weight_input_hidden*input_matrix)

o_hidden = ut.sigmoid_function(x_hidden)
# print(o_hidden)

x_output = np.matmul(weight_hidden_output, o_hidden)
# print(x_output)

output_matrix = ut.sigmoid_function(x_output)
# print(output_matrix)

error_matrix = np.array([[0.8], [0.5], [0.3]])
# print(error_matrix)

y_hidden = np.matmul(weight_hidden_output.T, error_matrix)
# print(y_hidden)

y_input = np.matmul(weight_input_hidden.T, y_hidden)
# print(y_input)

#######################################################################################
## derivative of error

delta_weight_hidden_output = np.matmul(-2*error_matrix*(ut.sigmoid_function(x_output))*(1-ut.sigmoid_function(x_output)), o_hidden.T)

print(weight_hidden_output)
print(delta_weight_hidden_output)

updated_weight_hidden_output = weight_hidden_output + delta_weight_hidden_output
print(updated_weight_hidden_output)


# updating weight matrix [ ouput & hidden layer ]