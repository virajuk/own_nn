import numpy as np

from .activation import Sigmoid


class NN:

    input_matrix = None
    weights = []
    no_of_nodes_in_layers = []
    activation_function = Sigmoid()

    def __init__(self, lr=0.01):
        self.learning_rate = lr

    def set_input_layer(self, input_matrix=np.array([])):

        self.input_matrix = input_matrix.flatten()
        self.no_of_nodes_in_layers.append(self.input_matrix.size)

    def add_hidden_layer(self, no_of_nodes=1):

        self.no_of_nodes_in_layers.append(no_of_nodes)
        self.weights.append(np.random.rand(self.no_of_nodes_in_layers[-1], self.no_of_nodes_in_layers[-2]))

    def set_output_layer(self, no_of_nodes=1):

        self.add_hidden_layer(no_of_nodes)

    def set_activation_function(self, activation):

        self.activation_function = activation()

    def model(self):

        return self

    def summary(self):

        print(self.no_of_nodes_in_layers)
        for weight in self.weights:

            # print("###########")
            print(weight.shape)
            # print(weight)

        print(self.activation_function.__repr__())
