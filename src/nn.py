import numpy as np

from .activation import Sigmoid


class NN:

    input_matrix = np.array([[]])
    weights = list()
    no_of_nodes_in_layers = list()

    def __init__(self, **kwargs):
        self.learning_rate = kwargs.get("lr", 0.01)
        self.activation_function = kwargs.get("activation", Sigmoid())

    def set_input_layer(self, input_matrix):

        self.input_matrix = input_matrix.flatten()
        self.no_of_nodes_in_layers.append(self.input_matrix.size)

    def add_hidden_layer(self, no_of_nodes=1):

        self.no_of_nodes_in_layers.append(no_of_nodes)
        self.weights.append(np.random.rand(self.no_of_nodes_in_layers[-1], self.no_of_nodes_in_layers[-2]))

    def set_output_layer(self, no_of_nodes=1):

        self.add_hidden_layer(no_of_nodes)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f" Learning_rate : {self.learning_rate}"
        repr_str += f" Activation : {self.activation_function}"
        repr_str += f" No of nodes in layers : {self.no_of_nodes_in_layers}"
        return repr_str
