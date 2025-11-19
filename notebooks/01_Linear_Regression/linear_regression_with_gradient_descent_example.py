import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.preprocessing import MinMaxScaler

def load_dataset_scaled():
    df = pd.read_csv('../../data/dataset_200x4_regression.csv')
    df = MinMaxScaler().fit_transform(df)
    X = df[:, :-1]
    t = df[:, -1]
    return X, t


def cost_f(X, t, weights):
    N = X.shape[0]
    predictions = X @ weights
    error = predictions - t
    cost = (error.T @ error) / (2*N)
    return cost

def f_derivative(X, t, weights):
    N = X.shape[0]
    predictions = X @ weights
    error = predictions - t
    gradient = (X.T @ error) / (N)
    return gradient

def gradient_descent_linear_regression(X, t, step_size=0.01, precision=0.0001, max_iter=1000):
    examples, features = X.shape
    no_iter = 0
    cur_weights = np.ones(features, dtype=np.float32)
    last_weights = np.array([np.inf for _ in range(len(cur_weights))])

    while norm(cur_weights - last_weights) > precision and no_iter < max_iter:
        print("weights: ", cur_weights)
        gradient = f_derivative(X, t, cur_weights)
        print(f"cost: {cost_f(X, t, cur_weights)} - gradient: {gradient}")

        last_weights = cur_weights.copy()
        cur_weights -= gradient * step_size
        no_iter += 1

    print(f'Optimal Cost: {cost_f(X, t, cur_weights)}')
    return cur_weights


if __name__ == '__main__':
    X, t = load_dataset_scaled()
    X = np.c_[np.ones((X.shape[0], 1)), X]
    # X = np.hstack([np.ones((X.shape[0], 1)), X])

    optimal_weights = gradient_descent_linear_regression(X, t,0.1,max_iter=3)