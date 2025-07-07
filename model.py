import numpy as np

# Hypothesis (prediction) function
def predict(x, theta0, theta1):
    return theta0 + theta1 * x

# Cost function (Mean Squared Error)
def compute_cost(x, y, theta0, theta1):
    m = len(y)
    predictions = predict(x, theta0, theta1)
    return (1 / (2 * m)) * ((predictions - y) ** 2).sum()

# Training function using gradient descent
def train_model(X, y, alpha=0.01, iterations=1000):
    X = np.array(X)
    mean = X.mean()
    std = X.std()
    X_norm = (X - mean) / std

    theta0 = 0
    theta1 = 0
    m = len(X)

    for i in range(iterations):
        predictions = predict(X_norm, theta0, theta1)
        error = predictions - y

        d_theta0 = (1 / m) * error.sum()
        d_theta1 = (1 / m) * (error * X_norm).sum()

        theta0 -= alpha * d_theta0
        theta1 -= alpha * d_theta1

        if i % 100 == 0:
            cost = compute_cost(X_norm, y, theta0, theta1)
            print(f"Iteration {i}: Cost = {cost:.4f}")

    return theta0, theta1, mean, std  # Return everything needed to use the model
