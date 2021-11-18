from src.first import First

data = [[3, 1, 0], [1, 3, 1], [3.1, 0.975, 0], [3.02, 0.99, 0], [1.03, 3.04, 1], [2.97, 0.99, 0], [1.09, 3.02, 1], [2.99, 1.01, 0], [1.1, 3.14, 1], [1.07, 2.97, 1],
        [3.07, 0.91, 0], [0.97, 2.94, 1], [3.01, 0.91, 0], [2.9, 0.94, 0], [3.1, 1.08, 0], [0.97, 2.99, 1], [3.11, 1.15, 0], [2.88, 0.89, 0], [0.93, 2.89, 1], [0.95, 2.92, 1]]

# print(len(data))

count = {}
for point in data:

    x, y, label = point
    if label not in count.keys():
        count[label] = 0

    count[label] += 1

first = First()
first.split_train_eval_data(data)

first.first_sample()
first.second_sample()

# print(first.check_point_above_classifier(6, 3))
