import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil


class Perceptron:

    def __init__(self, inputs, learning_data, learning_rate, epochs):
        self.inputs = inputs
        self.learning_data = learning_data
        self.learning_rate = learning_rate
        self.epochs = epochs  # number of iterations

    def activation(self, input):
        """
        Step function for the activation function
        """
        return 1.0 if input > 0 else 0.0

    def train(self):
        """
        Perceptron learning rule
        """

        _, features = self.inputs.shape

        # Initializing parapeters(theta) to zeros.
        # +1 in n+1 for the bias term.
        theta = np.zeros((features + 1, 1))

        # List to store how many examples were misclassified at every iteration.
        misclassified = []

        # Training.
        for _ in range(self.epochs):

            # counter to store misclassified.
            error_counter = 0

            # Looping for every example
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
                    error_counter += 1

            # Appending number of misclassified examples at every iteration.
            misclassified.append(error_counter)

        return theta, misclassified


def readFile(filename):
    """
    Read the file and return the data as a numpy array with inputs and desired outputs.
    """
    with open(f'../../data/{filename}.csv') as csvfile:
        readCSV = pd.read_csv(csvfile, skiprows=1,
                              header=None, delimiter=',').to_numpy()

        inputs = []
        desired = []
        for row in readCSV:
            inputs.append([row[0], row[1]])
            desired.append(row[2])

        return np.array(inputs), np.array(desired)


def clean_dataset(filename):
    """
    Clean the dataset to make it linearly separable
    """
    # Retrieving the original filename
    original_filename = filename.split('_', 1)[0]
    # Making a copy of the original file
    shutil.copy(f'../../data/{original_filename}.csv', f'../../data/{filename}.csv')

    df = pd.read_csv(f'../../data/{filename}.csv')

    # Update third column to 1 if first column is negative
    for index, row in df.iterrows():
        if row[0] < 0:
            df.at[index, 'y'] = 1
        else:
            df.at[index, 'y'] = 0

    # Save to csv
    df.to_csv('../../data/dataset_clean.csv', index=False)


def plot_decision_boundary(inputs, y, theta, filename):

    # Equation ax+b = theta0.X0 + theta1.X1 + theta2.X2
    a = -theta[1]/theta[2]
    b = -theta[0]/theta[2]
    x1 = [min(inputs[:, 0]), max(inputs[:, 0])]
    x2 = a * x1 + b

    # Plotting the decision boundary
    plt.figure(figsize=(10, 8))
    plt.title('Perceptron Decision Boundary')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.plot(inputs[:, 0][y == 0], inputs[:, 1][y == 0], "o", color="red")
    plt.plot(inputs[:, 0][y == 1], inputs[:, 1][y == 1], "o", color="blue")
    plt.plot(x1, x2, color='black')
    plt.savefig(f'../../images/{filename}.png')
    plt.show()


if __name__ == '__main__':
    # Dataset filename to be used
    filename = sys.argv[1]

    # Clean dataset to make it linearly separable
    if 'clean' in filename:
        clean_dataset(filename)

    # Reading the dataset and formatting it
    inputs, desired = readFile(filename)

    # Initializing the Perceptron
    perceptron = Perceptron(inputs, desired, 0.5, 100)

    # Training the Perceptron
    theta, misclassified = perceptron.train()

    # Plotting the decision boundary
    plot_decision_boundary(inputs, desired, theta, filename)
