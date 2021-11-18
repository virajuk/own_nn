import numpy as np
import random

random.seed(13)


class First:

    train_data, test_data = None, None
    slope = 0

    def __init__(self):
        self.slope = random.random()

    def split_train_eval_data(self, data, ratio=0.8):

        train_data_length = int(len(data)*ratio)
        test_data_length = len(data) - train_data_length

        self.train_data = data[:train_data_length]
        self.test_data = data[-test_data_length:]

    def calculate_desired_target(self, y, actual_output):
        b = random.random()

        if y > actual_output:
            desired_target = y + b
        else:
            desired_target = y - b
        return desired_target

    def first_sample(self):

        x, y, label = self.train_data[0]
        message = f"Slope : {self.slope} \t y : {y} \t x : {x} \t"

        actual_output = self.slope * x
        message += f"Actual output : {actual_output} \t"

        desired_target = self.calculate_desired_target(y, actual_output)
        message += f"Desired target : {desired_target} \t"

        print(message)
