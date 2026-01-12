import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import norm
from .cost import compute_cost, compute_derivative
from gradient_descent.gradient_check import gradient_check

def fit_gd(X, y, alpha=0.01, precision=0.0001, max_iter=1000):
    """
    Perform linear regression using gradient descent.

    Parameters:
    X : numpy.ndarray
        Feature matrix.
    y : numpy.ndarray
        Target vector.
    alpha : float
        Learning rate.
    precision : float
        Precision for convergence criteria.
    max_iter : int
        Maximum number of iterations.
    Returns:
    theta : numpy.ndarray
        Estimated parameters.
    """
    
    def grad_wrapper(current_params):
        return compute_derivative(X, y, current_params)
    
    def cost_wrapper(current_params):
        return compute_cost(X, y, current_params)

    n = X.shape[1] # Number of features including bias term
    curr_values = np.array([0.0 for _ in range(n)])
    last_values = np.array([np.inf for _ in range(len(curr_values))])

    number_iterations = 0
    cost_values = []
    while number_iterations < max_iter and norm(curr_values - last_values) > precision:
        last_values = curr_values.copy()
        curr_values -= alpha * grad_wrapper(curr_values)
        number_iterations += 1
        cost = cost_wrapper(curr_values)
        # print(f"Iteration {number_iterations}: Current values = {curr_values}, Cost = {cost}")
        cost_values.append(cost)
        

    return curr_values, cost_values

def fit_normal_eq(X, y):
    """
    Compute the parameters for linear regression using the normal equation.

    Parameters:
    X : numpy.ndarray
        The input feature matrix (m x n) where m is the number of samples and n is the number of features.
    y : numpy.ndarray
        The output target vector (m x 1).

    Returns:
    numpy.ndarray
        The computed parameters (n x 1).
    """

    # Add a column of ones to X to account for the intercept term
    X_b = np.c_[np.ones((X.shape[0], 1)), X]

    # Compute the parameters using the normal equation
    best_weights = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    return best_weights

def predict(X, weights):
    """
    Predict the target values using the linear regression parameters.

    Parameters:
    X : numpy.ndarray
        Feature matrix.
    params : numpy.ndarray
        Estimated parameters.

    Returns:
    numpy.ndarray
        Predicted target values.
    """
    return X.dot(weights)

# Example usage and verification
def main():
    X = np.array([0, 0.2, 0.4, 0.8, 1.0])
    t = 5+X
    
    X = X.reshape(-1, 1)  # Reshape for a single feature
    X_b = np.c_[np.ones(X.shape[0]), X]  # Add bias term

    best_params, _ = fit_gd(X=X_b, y=t, alpha=0.1, precision=0.000001, max_iter=1000)
    print("Optimal parameters (bias and weights):", best_params)
    # Expected output is approximately [5, 1] for bias and weight respectively.
    
    # Visualization Verification
    plt.scatter(X, t, color='blue', label='Data points')
    plt.plot(X, X_b.dot(best_params), color='red', label='Fitted line')
    plt.xlabel('X')
    plt.ylabel('t')
    plt.title('Linear Regression with Gradient Descent')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()