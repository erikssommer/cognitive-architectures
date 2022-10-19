# Load math library
import numpy as np

# Load plotting library
import matplotlib.pyplot as plt

# Function that creates a series of numbers using chaos theory
def logistic_map(x0: float, n: int) -> list:
    x_values = []
    r = 3.9 # Rate, (The Edge of Chaos (r ≈ 3.57))
    x = x0 # Initial value
    
    # Generate all values of x given the initial value
    for i in range(n):
        x_values += [x] # Add the number to the list
        x = x * r * (1 - x) # Logistic map equation
    
    # Return x and y
    return np.arange(len(x_values)), x_values

# Function that plot a (x, y) series of points
def plot_line_chart(x, y, x_label, y_label, title):
    plt.figure(figsize = (16, 4))
    plt.plot(x, y, label = y_label)
    plt.xlabel(x_label, fontsize = 11)
    plt.ylabel(y_label, fontsize = 11)
    plt.title(title, size=14)
    plt.legend(loc = 'upper right')
    plt.show()

# Create and plot 100 random-chaos numbers
x0 = 0.7
x, y = logistic_map(x0, 100)
plot_line_chart(x, y, '$x$', '$chaos(x)$', 'Plot created with the logistic map chaos function')