import numpy as np

def gradient_check(weights, compute_cost, compute_derivative, epsilon=1e-6):
    """
    Perform gradient checking to verify the correctness of the derivative computation.
    """
    analytical_grad = compute_derivative(weights)

    for i in range(len(weights)):
        weights_plus = np.copy(weights)
        weights_minus = np.copy(weights)

        weights_plus[i] += epsilon
        weights_minus[i] -= epsilon

        J_plus = compute_cost(weights_plus)
        J_minus = compute_cost(weights_minus)

        grad_numerical = (J_plus - J_minus) / (2 * epsilon)
        grad_analytical = analytical_grad[i]

        diff = abs(grad_numerical - grad_analytical) / max(1.0, abs(grad_numerical) + abs(grad_analytical))

        if diff > 1e-7:
            print(f"Gradient check failed at index {i}. Numerical: {grad_numerical}, Analytical: {grad_analytical}, Diff: {diff}")
            return diff

    return 0.0