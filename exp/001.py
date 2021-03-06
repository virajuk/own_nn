import random


def calculate_error(desired_target, actual_output):

    err = desired_target - actual_output
    return err


# (x, y) co-ordinates
training_data = [[3, 1, 0], [1, 3, 1], [3.1, 0.975, 0]]
eval_data = [[3.02, 0.99, 0], [1.03, 3.04, 1]]

# since it's linear y = ax
a, x = 0, 0
y = a*x

random.seed(13)
a = random.random()
# print(a)


def eval_results():

    pass


def first_sample(a, training_data):

    x, y, label = training_data[0]
    message = f"A : {a} \t y : {y} \t x : {x} \t"

    actual_output = a*x
    message += f"Actual output : {actual_output} \t"

    desired_target = calculate_desired_target(y)
    message += f"Desired target : {desired_target} \t"

    error = calculate_error(desired_target, actual_output)
    message += f"Error : {error} \t"

    print(message)


def calculate_desired_target(y):

    b = random.random()
    desired_target = y + b
    return desired_target

first_sample(a, training_data)
