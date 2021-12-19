import numpy as np
from utils import utils as ut


def __calculate_error(label, output_output):

    error = (label - output_output)**2
    return error


LABEL = np.array([[1.0], [0.0]])

input_input = np.array([[3.0], [1.0]])
weight_input_hidden = np.array([[8.0, 3.0], [6.0, 5.0]])
weight_hidden_output = np.array([[1.0, 0.7], [1.6, 4.3]])

input_hidden = np.matmul(weight_input_hidden, input_input)
output_hidden = ut.sigmoid_function(input_hidden)

input_output = np.matmul(weight_hidden_output, output_hidden)
output_output = ut.sigmoid_function(input_output)

print(output_output)
error = __calculate_error(LABEL, output_output)

print(error)
