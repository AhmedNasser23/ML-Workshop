import numpy as np
import matplotlib.pyplot as plt
from utils.dataset_loader import add_bias_term

def visualize_cost_convergence(cost_values):
    plt.plot(np.arange(len(cost_values)), cost_values, color='blue')
    plt.title('Cost Function Convergence')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cost Value')
    plt.grid()
    plt.show()

def visualize_each_feature_vs_target(X, y):
    num_features = X.shape[1]
    for i in range(1, num_features):  # Skip bias term at index 0
        plt.figure()
        plt.scatter(X[:, i], y, color='blue', label='Data Points')
        plt.title(f'Feature {i} vs Target Variable')
        plt.xlabel(f'Feature {i}')
        plt.ylabel('Target Variable')
        plt.grid()
        plt.legend()
        plt.show()

def visualize_single_feature_regression(X, y, w):
    plt.scatter(X, y, color='blue', label='Data')

    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    X_line_b = add_bias_term(X_line)
    y_pred = X_line_b @ w

    plt.plot(X_line, y_pred, color='red', label='Regression Line')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.title('Single Feature Linear Regression')
    plt.legend()
    plt.grid()
    plt.show()
