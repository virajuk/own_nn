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

    def calculate_desired_target(self, x, y, label):
        b = random.random()

        if self.check_point_above_classifier(x, y):
            if label:
                desired_target = y - b
            else:
                desired_target = y + b
        else:
            if label:
                desired_target = y + b
            else:
                desired_target = y - b

        return desired_target

    def calculate_error(self, desired_target, actual_output):

        err = desired_target - actual_output
        return err

    def calculate_increment(self, x, error):

        delta = error/x
        self.slope += delta
        return delta

    def check_point_above_classifier(self, x, y):

        if (y - self.slope*x) > 0:
            return True
        else:
            return False

    def first_sample(self):

        x, y, label = self.train_data[0]
        message = f"Slope : {self.slope} \t y : {y} \t x : {x} \t label : {label} \t"

        actual_output = self.slope * x
        message += f"Actual output : {actual_output} \t"

        desired_target = self.calculate_desired_target(x, y, label)
        message += f"Desired target : {desired_target} \t"

        error = self.calculate_error(desired_target, actual_output)
        message += f"Error : {error} \t"

        delta = self.calculate_increment(x, error)
        message += f"Delta : {delta} \t"

        print(message)

    def second_sample(self):

        x, y, label = self.train_data[1]
        message = f"Slope : {self.slope} \t y : {y} \t x : {x} \t label : {label} \t"

        actual_output = self.slope * x
        message += f"Actual output : {actual_output} \t"

        desired_target = self.calculate_desired_target(x, y, label)
        message += f"Desired target : {desired_target} \t"

        error = self.calculate_error(desired_target, actual_output)
        message += f"Error : {error} \t"

        delta = self.calculate_increment(x, error)
        message += f"Delta : {delta} \t"

        print(message)

    def train_model(self):

        count = 0
        for data_point in self.train_data:

            x, y, label = data_point
            message = f"Slope : {self.slope} \t y : {y} \t x : {x} \t label : {label} \t"

            actual_output = self.slope * x
            message += f"Actual output : {actual_output} \t"

            desired_target = self.calculate_desired_target(x, y, label)
            message += f"Desired target : {desired_target} \t"

            error = self.calculate_error(desired_target, actual_output)
            message += f"Error : {error} \t"

            delta = self.calculate_increment(x, error)
            message += f"Delta : {delta} \t"

            print(message)

            count += 1
            # if count == 3:
            #     break

