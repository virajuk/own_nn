# 3 layer 2 node network

import time
import numpy as np

from utils import utils as ut

input_input = np.array([[3.0], [1.0]])
weight_input_hidden = np.array([[8.0, 3.0], [6.0, 5.0]])
weight_hidden_output = np.array([[1.0, 0.7], [1.6, 4.3]])

LABEL = np.array([[1.0], [0.0]])
LEARNING_RATE = 0.1


def single_iteration(weight_hidden_output, weight_input_hidden):

    # print(f" Weight Input Hidden : \n {weight_input_hidden} \n Weight Hidden Output : \n {weight_hidden_output}")

    ### forward pass

    # input & output of hidden layer
    input_hidden = np.matmul(weight_input_hidden, input_input)
    output_hidden = ut.sigmoid_function(input_hidden)

    # input & output of output layer
    input_output = np.matmul(weight_hidden_output, output_hidden)
    output_output = ut.sigmoid_function(input_output)

    ### error calculation of the output layer

    error = __calculate_error(LABEL, output_output)
    # print(error)

    ### error back propagation

    # error hidden layer
    error_hidden = np.matmul(weight_hidden_output.T, error)
    # print(error_hidden)

    ### calculate delta for weights
    delta_weight_hidden_output = np.matmul(LEARNING_RATE*(-2)*error*(ut.sigmoid_function(input_output))*(1-ut.sigmoid_function(input_output)), output_hidden.T)
    # print(delta_weight_hidden_output)

    weight_hidden_output = weight_hidden_output + delta_weight_hidden_output
    # print(new_weight_hidden_output)

    # error_input = np.matmul(weight_input_hidden.T, error_hidden)

    delta_weight_input_hidden = np.matmul(LEARNING_RATE*(-2)*error_hidden*(ut.sigmoid_function(input_hidden))*(1-ut.sigmoid_function(input_hidden)), input_hidden.T)

    weight_input_hidden = weight_input_hidden + delta_weight_input_hidden

    # print(f" Updated Weight Input Hidden : \n {weight_input_hidden} \n Updated Weight Hidden Output : \n {weight_hidden_output}")

    return weight_input_hidden, weight_hidden_output, error

def __calculate_error(label, output_output):

    error = (label - output_output)**2
    return error


# single_iteration(weight_hidden_output, weight_input_hidden)

for i in range(10000):

    weight_input_hidden, weight_hidden_output, error = single_iteration(weight_hidden_output, weight_input_hidden)

    if i % 100 == 0:
        message = f"Iter : {i} \n"
        # message += f"Updated Weight Input Hidden : \n {weight_input_hidden} \n Updated Weight Hidden Output : \n {weight_hidden_output} \n"
        message += f"Error : {error} \n"

        f = open("logs/logs.txt", "a+")
        f.write(message)
        f.close()

    # time.sleep(2)
