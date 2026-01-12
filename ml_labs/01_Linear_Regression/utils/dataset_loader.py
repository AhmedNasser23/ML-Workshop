import pandas as pd
import numpy as np

def load_dataset(file_path):
    """
    Load dataset from a CSV file.

    Parameters:
    file_path : str
        Path to the CSV file.

    Returns:
    X : numpy.ndarray
        Feature matrix.
    y : numpy.ndarray
        Target vector.
    """
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1].values  # All columns except the last one as features
    y = data.iloc[:, -1].values   # Last column as target
    return X, y

def add_bias_term(X):
    """
    Add a bias term (column of ones) to the feature matrix.

    Parameters:
    X : numpy.ndarray
        Original feature matrix.

    Returns:
    X_b : numpy.ndarray
        Feature matrix with bias term added.
    """
    return np.c_[np.ones(X.shape[0]), X]  # Add a column of ones at the beginning

# Example usage:
def main():
    X, y = load_dataset('dataset_200x4_regression.csv')
    X_b = add_bias_term(X)
    print("Feature matrix with bias term:\n", X_b)
    print("Target vector:\n", y)

if __name__ == "__main__":
    main()