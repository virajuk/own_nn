import numpy as np
import random

random.seed(13)


class First:

    train_data, test_data = None, None
    slope = 0
    message = ""

    def __init__(self):
        self.slope = random.random()

    def split_train_eval_data(self, data, ratio=0.8):

        train_data_length = int(len(data)*ratio)
        test_data_length = len(data) - train_data_length

        self.train_data = data[:train_data_length]
        self.test_data = data[-test_data_length:]

    def calculate_desired_target(self, x, y, label):
        b = random.random()

        self.message += f"b : {format(b, '.4f')} \t"

        if self.check_point_above_classifier(x, y):
            self.message += f" Above : True \t"
            if label:
                desired_target = y
            else:
                desired_target = y + b
        else:
            self.message += f" Above : False \t"
            if label:
                desired_target = y - b
            else:
                desired_target = y

        return desired_target

    def calculate_error(self, desired_target, actual_output):

        err = desired_target - actual_output
        return err

    def calculate_increment(self, x, error, lr=0.5):

        delta = lr*(error/x)
        self.slope += delta
        return delta

    def check_point_above_classifier(self, x, y):

        if (y - self.slope*x) > 0:
            return True
        else:
            return False

    def train_model(self):

        count = 0
        for data_point in self.train_data:

            x, y, label = data_point
            self.message = f"Slope : {format(self.slope, '.4f')} \t y : {format(y, '.2f')} \t x : {format(x, '.2f')} \t label : {label} \t"

            actual_output = self.slope * x
            self.message += f"Actual output : {format(actual_output, '.4f')} \t"

            desired_target = self.calculate_desired_target(x, y, label)
            self.message += f"Desired target : {format(desired_target, '.4f')} \t"

            error = self.calculate_error(desired_target, actual_output)
            self.message += f"Error : {format(error, '.4f')} \t"

            delta = self.calculate_increment(x, error, 0.1)
            self.message += f"Delta : {format(delta, '.4f')} \t"

            print(self.message)

            count += 1
            # if count == 3:
            #     break


