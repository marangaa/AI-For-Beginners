# Import necessary libraries
import numpy as np

# Define input data and target labels
x = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
t = np.array([0, 1, 1, 1])

# Initialize weights and bias
weights = np.random.rand(2)  # Two input features
bias = np.random.rand()

# Define the perceptron function
def perceptron(input_data, weights, bias):
    activation = np.dot(weights, input_data) + bias
    return 1.0 if activation >= 0.0 else 0.0

# Train the perceptron
for _ in range(100):  # Number of epochs
    for i in range(len(t)):
        prediction = perceptron(x[:, i], weights, bias)
        error = t[i] - prediction
        weights += error * x[:, i]
        bias += error

# Make predictions
for i in range(len(t)):
    print(f"Input: {x[:, i]}, Prediction: {perceptron(x[:, i], weights, bias)}")
