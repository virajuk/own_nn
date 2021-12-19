import numpy as np


class NN:

    input_matrix = None
    weights = []
    no_of_nodes_in_layers = []

    def __init__(self, lr=0.01):
        self.learning_rate = lr

    def input_layer(self, input_matrix=np.array([])):

        self.input_matrix = input_matrix.flatten()
        self.no_of_nodes_in_layers.append(self.input_matrix.size)

    def add_hidden_layer(self, no_of_nodes=1):

        self.no_of_nodes_in_layers.append(no_of_nodes)
        self.weights.append(np.random.rand(self.no_of_nodes_in_layers[-1], self.no_of_nodes_in_layers[-2]))

    def output_layer(self, no_of_nodes=1):

        self.add_hidden_layer(no_of_nodes)

    def summary(self):

        print(self.no_of_nodes_in_layers)
        for weight in self.weights:

            print("###########")
            print(weight.shape)
            print(weight)

