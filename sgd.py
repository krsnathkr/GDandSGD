import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('values.txt', delimiter='\t', skiprows=1)  

X = data[:, :-1]
y = data[:, -1] 


X = np.c_[np.ones(X.shape[0]), X]


np.random.seed(0)  # For reproducibility
weights = np.random.randn(X.shape[1])

# Hyperparameters
learning_rate = 0.01
epochs = 100

# Function to calculate mean squared error
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Stochastic Gradient Descent
history_weights = []  # Store history of weights for each epoch
losses = []           # Store history of losses for each epoch

for epoch in range(epochs):
    epoch_loss = 0
    for i in range(X.shape[0]):
        # Randomly select a data point
        random_index = np.random.randint(0, X.shape[0])
        x_i = X[random_index]
        y_i = y[random_index]

        # Compute prediction and error
        prediction = np.dot(x_i, weights)
        error = prediction - y_i

        # Update weights
        weights -= learning_rate * error * x_i

        # Track loss for this data point
        epoch_loss += error ** 2
    
    # Track history for this epoch
    history_weights.append(weights.copy())
    losses.append(epoch_loss / X.shape[0])

    # Plot updated fitting curve for each epoch
    if epoch % 10 == 0:
        plt.scatter(range(X.shape[0]), y, color='blue', label='Actual')
        plt.scatter(range(X.shape[0]), np.dot(X, weights), color='red', label='Predicted')
        plt.title(f'Epoch {epoch}')
        plt.xlabel('Data Point')
        plt.ylabel('Target')
        plt.legend()
        plt.show()

# Report history of feature weights update for every epoch
for i, w in enumerate(history_weights):
    print(f'Epoch {i}: {w}')

# Plot the loss curve
plt.plot(range(epochs), losses)
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.show()

#print the final weights and loss
print(f'Final Weights: {weights}')
print(f'Final Loss: {losses[-1]}')

print(len(weights))
