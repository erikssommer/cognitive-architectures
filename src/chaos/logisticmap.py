# Load math library
import numpy as np
# Load system library
import sys
# Load plotting library
import matplotlib.pyplot as plt


def logistic_map(x0: float, rate: float, n: int) -> tuple:
    """
    Function that creates a series of numbers using chaos theory
    """
    x_values = []
    r = rate  # Rate, (The Edge of Chaos (r ≈ 3.57))
    x = x0  # Initial value

    # Generate all values of x given the initial value
    for i in range(n):
        x_values += [x]  # Add the number to the list
        x = x * r * (1 - x)  # Logistic map equation

    # Return x and y
    return np.arange(len(x_values)), x_values


def plot_line_chart(x: int, y: float, x_label: str, y_label: str, title: str) -> None:
    """
    Function that plot a (x, y) series of points
    """
    plt.figure(figsize=(10, 4))
    plt.plot(x, y, label=y_label)
    plt.xlabel(x_label, fontsize=10)
    plt.ylabel(y_label, fontsize=10)
    plt.title(title, size=14)
    plt.legend(loc='upper right')
    plt.show()


# Create and plot 100 random-chaos numbers given initial value and rate from shell input
init_value_x = sys.argv[1]
x0 = float(init_value_x)
rate = sys.argv[2]
r = float(rate)
x, y = logistic_map(x0, r, 100)
plot_line_chart(x, y, '$x$', '$chaos(x)$', 'Initial value: {}, Rate: {}'.format(x0, rate))
