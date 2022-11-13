import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Perceptron:

    def __init__(self, inputs, learning_data, learning_rate, epochs):
        self.inputs = inputs
        self.learning_data = learning_data
        self.learning_rate = learning_rate
        self.epochs = epochs  # number of iterations

    def activation(self, input):
        return 1.0 if input > 0 else 0.0

    def train(self):

        _, features = self.inputs.shape

        # Initializing parapeters(theta) to zeros.
        # +1 in n+1 for the bias term.
        theta = np.zeros((features + 1, 1))

        # Empty list to store how many examples were
        # misclassified at every iteration.
        n_miss_list = []

        # Training.
        for _ in range(self.epochs):

            # counter to store misclassified.
            n_miss = 0

            # looping for every example
            for idx, x_i in enumerate(self.inputs):

                # Insering 1 for bias, X0 = 1
                x_i = np.insert(x_i, 0, 1).reshape(-1, 1)

                # Calculating prediction/hypothesis
                y_hat = self.activation(np.dot(x_i.T, theta))

                # Updating if the example is misclassified
                if (np.squeeze(y_hat) - self.learning_data[idx]) != 0:
                    theta += self.learning_rate * \
                        ((self.learning_data[idx] - y_hat)*x_i)

                    # Incrementing by 1
                    n_miss += 1

            # Appending number of misclassified examples
            # at every iteration.
            n_miss_list.append(n_miss)

        return theta, n_miss_list


def readFiles(filename):
    with open(f'../../data/{filename}.csv') as csvfile:
        readCSV = pd.read_csv(csvfile, skiprows=1,
                              header=None, delimiter=',').to_numpy()

        inputs = []
        desired = []
        for row in readCSV:
            inputs.append([row[0], row[1]])
            desired.append(row[2])

        return np.array(inputs), np.array(desired)


def plot_decision_boundary(inputs, y, theta, filename):

    # The Line is y=ax+b
    # So, Equate ax+b = theta0.X0 + theta1.X1 + theta2.X2
    # Solving we find m and c
    x1 = [min(inputs[:, 0]), max(inputs[:, 0])]
    a = -theta[1]/theta[2]
    b = -theta[0]/theta[2]
    x2 = a*x1 + b

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.plot(inputs[:, 0][y == 0], inputs[:, 1][y == 0], "r^")
    plt.plot(inputs[:, 0][y == 1], inputs[:, 1][y == 1], "bs")
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.title('Perceptron Algorithm')
    plt.plot(x1, x2, 'y-')
    plt.savefig(f'../../images/{filename}.png')
    plt.show()


if __name__ == '__main__':
    filename = sys.argv[1]

    inputs, desired = readFiles(filename)

    perceptron = Perceptron(inputs, desired, 0.5, 100)

    theta, misclassified = perceptron.train()

    plot_decision_boundary(inputs, desired, theta, filename)
