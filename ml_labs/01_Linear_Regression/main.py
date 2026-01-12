import numpy as np
import argparse
from sklearn.linear_model import LinearRegression

from linear_regression.linear_regression import fit_gd, fit_normal_eq
from linear_regression.cost import compute_cost, compute_derivative
from utils.dataset_loader import load_dataset, add_bias_term
from utils.scaling import standard_scale_data, minmax_scale_data
from utils.config_loader import load_config
from gradient_descent.gradient_check import gradient_check

# ---- Argument Parsing ----
def parse_arguments():
    parser = argparse.ArgumentParser(description="Linear Regression Practices")

    parser.add_argument('--config', type=str, help='Path to the configuration file (JSON or YAML)')

    parser.add_argument('--dataset', type=str, default='data/dataset_200x4_regression.csv', help='Path to the dataset file')


    parser.add_argument('--preprocessing', 
                        type=int, 
                        choices=[0, 1, 2],
                        help=(
                            'Preprocessing: '
                            '0=none, '
                            '1=min-max scaling, '
                            '2=standardization'
                        )
                    )

    parser.add_argument('--choice', 
                        type=int, 
                        choices=[0, 1, 2, 3, 4],
                        help=(
                        'Operation mode: '
                        '0=linear verification, '
                        '1=gradient descent, '
                        '2=normal equation, '
                        '3=scikit-learn'
                        )
    )

    parser.add_argument('--grad-check', action='store_true', help='Run gradient checking before training (debug only)')

    parser.add_argument('--alpha', type=float, help='Learning rate for gradient descent')

    parser.add_argument('--precision', type=float, help='Precision for convergence criteria')

    parser.add_argument('--max_iter', type=int, help='Maximum number of iterations for gradient descent')

    return parser.parse_args()

# ---- Argument validation ----
def validate_arguments(args):
    gd_modes = {0, 1}
    if args.choice not in gd_modes:
        if any(v is not None for v in [args.alpha, args.precision, args.max_iter]):
            raise ValueError(
                "Arguments --alpha, --precision, and --max_iter are only valid for "
                "gradient descent modes (choice 0 or 1)."
            )
    if args.choice == 2 and args.preprocessing != 0:
        raise ValueError(
            "Preprocessing is not applicable when using the normal equation (choice 2), use --preprocessing 0."
        )

# ---- Config merging ----
def merge_config_args(config: dict, args: argparse.Namespace) -> argparse.Namespace:
    for key, value in config.items():
        # if the argument was not provided via command line, use the config value
        if hasattr(args, key) and getattr(args, key) is None:
            setattr(args, key, value)
    return args

# ---- Main execution ----
def main():
    args = parse_arguments()

    # Load config file if provided
    if args.config:
        config = load_config(args.config)
        args = merge_config_args(config, args)

    validate_arguments(args)

    # print(vars(args))

    # Load and preprocess dataset
    X, y = load_dataset(args.dataset)

    if args.preprocessing == 1:
        X = minmax_scale_data(X)
    elif args.preprocessing == 2:
        X = standard_scale_data(X)

    if args.choice in {0, 1, 2}:
        X_b = add_bias_term(X)

    if args.grad_check and args.choice in {0, 1}:
        def cost_function(weights):
            return compute_cost(X_b, y, weights)
        def gradient_function(weights):
            return compute_derivative(X_b, y, weights)
        
        # Gradient checking
        weights_test = np.random.randn(X_b.shape[1]) * 0.01
        diff = gradient_check(weights=weights_test, compute_cost=cost_function, compute_derivative=gradient_function)
        print(f"Gradient check difference: {diff}")
        if diff < 1e-7:
            print("Gradient check passed!")
        else:
            print("Gradient check failed!")
        return  # Exit after gradient check

    if args.choice == 0:
        # Linear verification
        weights_gd, _ = fit_gd(X_b, y, args.alpha, args.precision, args.max_iter)
        try:
            weights_ne = fit_normal_eq(X_b, y)
        except np.linalg.LinAlgError:
            weights_ne = None
            print("Normal Equation failed due to singular matrix.")
        print("Weights from Gradient Descent:", weights_gd)
        print("Cost using GD weights:", compute_cost(X_b, y, weights_gd))
        if weights_ne is not None:
            print("Weights from Normal Equation:", weights_ne)
            print("Cost using NE weights:", compute_cost(X_b, y, weights_ne))

    elif args.choice == 1:
        # Gradient Descent
        best_weights, _ = fit_gd(X_b, y, args.alpha, args.precision, args.max_iter)
        print("Best weights using all features:", best_weights)
        print("Cost using best weights:", compute_cost(X_b, y, best_weights))

    elif args.choice == 2:
        # Normal equation
        try:
            weights_ne = fit_normal_eq(X_b, y)
            print("Weights from Normal Equation:", weights_ne)
            print("Cost using NE weights:", compute_cost(X_b, y, weights_ne))
        except np.linalg.LinAlgError:
            print("Normal Equation failed due to singular matrix.")

    elif args.choice == 3:
        # Scikit-learn implementation
        model = LinearRegression()
        model.fit(X, y)
        print("Weights from Scikit-learn Linear Regression:", model.coef_, model.intercept_)
        print("Cost using Scikit-learn weights:", compute_cost(np.c_[np.ones(X.shape[0]), X], y, np.concatenate(([model.intercept_], model.coef_)) ))

if __name__ == "__main__":
    main()