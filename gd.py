import numpy as np
import matplotlib.pyplot as plt

def read_data(filename):
  with open(filename, 'r') as f:
    # Skip header
    next(f)
    lines = f.readlines()
  # Split each line by tab, remove leading/trailing whitespaces, and convert to float
  x = np.array([[float(val.strip()) for val in line.strip().split('\t')] for line in lines])[:, :-1]
  y = np.array([float(line.strip().split('\t')[-1]) for line in lines])
  return x, y

def gradient_descent(x, y, learning_rate=0.01, epochs=100):
  """
  Performs gradient descent to update feature weights.

  Args:
      x: A NumPy array of features (shape: n_samples, n_features).
      y: A NumPy array of target values (shape: n_samples).
      learning_rate: The learning rate for weight updates.
      epochs: The number of epochs for training.

  Returns:
      A tuple containing two lists:
          - weight_history: A list of lists containing weights for each epoch.
          - loss_history: A list containing the loss value for each epoch.
  """
  # Add bias term (constant feature with value 1)
  x = np.hstack((np.ones((x.shape[0], 1)), x))
  n_samples, n_features = x.shape

  # Initialize weights with random values
  weights = np.random.rand(n_features)

  weight_history = []
  loss_history = []
  for epoch in range(epochs):
    # ... rest of the code remains the same ...


    # Calculate predicted values
    predicted_y = np.dot(x, weights)

    # Calculate loss (mean squared error)
    loss = np.mean((predicted_y - y) ** 2)
    loss_history.append(loss)

    # Calculate gradient of loss w.r.t. weights
    gradient = 2 * np.dot(x.T, (predicted_y - y)) / n_samples

    # Update weights
    weights -= learning_rate * gradient

    weight_history.append(weights.copy())

    # Print loss for every 10 epochs
    if epoch % 10 == 0:
      print(f"Epoch: {epoch+1}, Loss: {loss:.4f}")

  return weight_history, loss_history

def plot_data_and_fit(x, y, weights, epoch):
  """
  Plots the data points and the fitting line for a given epoch.

  Args:
      x: A NumPy array of features (shape: n_samples, n_features).
      y: A NumPy array of target values.
      weights: A NumPy array of weights for the current epoch.
      epoch: The current epoch number.
  """
  # Add bias term (constant feature with value 1)
  x = np.hstack((np.ones((x.shape[0], 1)), x))
  predicted_y = np.dot(x, weights)
  plt.scatter(x[:, 1], y)  # Assuming first feature is the bias term, second is first actual feature
  plt.plot(x[:, 1], predicted_y, label=f"Epoch {epoch+1}")
  plt.legend()
  plt.show()



# Read data from text file
x, y = read_data("values.txt")

# Perform gradient descent
weight_history, loss_history = gradient_descent(x, y)

# Print final weights
print("\nFinal Weights:", weight_history[-1])

# Plot data points and fitting curves for some epochs
for epoch in [0, 24, 49, 99]:
  plot_data_and_fit(x, y, weight_history[epoch], epoch)

# Plot loss history (optional)
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()