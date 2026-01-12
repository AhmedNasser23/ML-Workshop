import numpy as np

def compute_cost(X, y, weights):
    """
    Compute the cost function for linear regression.

    Parameters:
    X : array-like, shape (m, n)
        The input feature matrix where m is the number of samples and n is the number of features.
    y : array-like, shape (m,)
        The output target vector.
    weights : array-like, shape (n,)
        The weights (parameters) for the linear regression model.

    Returns:
    float
        The computed cost.
    """
    N = X.shape[0]  # Number of samples
    prediction = X @ weights  # Predicted values
    error = prediction - y  # Prediction error
    cost = (1 / (2 * N)) * (error.T @ error)  # Cost calculation
    return cost

def compute_derivative(X, y, weights):
    """
    Compute the derivative of the cost function for linear regression.

    Parameters:
    X : array-like, shape (m, n)
        The input feature matrix where m is the number of samples and n is the number of features.
    y : array-like, shape (m,)
        The output target vector.
    weights : array-like, shape (n,)
        The weights (parameters) for the linear regression model.

    Returns:
    array-like, shape (n,)
        The derivative of the cost with respect to the weights.
    """
    N = X.shape[0]  # Number of samples
    prediction = X @ weights  # Predicted values
    error = prediction - y  # Prediction error
    derivative = (1 / N) * (X.T @ error)  # Derivative calculation
    return derivative

def main():
    
    # Example usage
    X = np.array([0, 0.2, 0.4, 0.8, 1.0])
    t = 5+X
    
    X = X.reshape(-1, 1)  # Reshape for a single feature
    X_b = np.c_[np.ones(X.shape[0]), X]  # Add bias term
    weights = np.array([1.0, 1.0])  # Initial weights
    cost = compute_cost(X_b, t, weights)
    derivative = compute_derivative(X_b, t, weights)
    print("Cost:", cost)
    print("Cost Derivative:", derivative)

if __name__ == "__main__":
    main()