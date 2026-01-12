import numpy as np
from numpy.linalg import norm

def gradient_descent(function_derivative, initial_start, step_size = 0.001, precision = 0.00001, max_iter = 1000):
    """Gradient Descent Algorithm
    Args:
        function_derivative: function that computes the derivative of the function to minimize
        initial_start: initial point to start the search
        step_size: step size for each iteration
        precision: precision to determine convergence
        max_iter: maximum number of iterations
    Returns:
        The point that minimizes the function
    """
    
    curr_values = initial_start
    last_values = np.array([np.inf for _ in range(len(initial_start))])

    number_iterations = 0
    while number_iterations < max_iter and norm(curr_values - last_values) > precision:
        last_values = curr_values.copy()
        curr_values -= step_size * function_derivative(curr_values)
        number_iterations += 1

    return curr_values

def trial1():
    def f(x, y):
        return 3 * (x + 2) ** 2 + (y - 1) ** 2       # 3(x + 2)² + (y - 1)²

    def function_derivative_dx(x):
        return 6 * (x + 2)

    def function_derivative_dy(y):
        return 2 * (y - 1)

    def function_derivative(states):
        x, y = states[0], states[1]
        return np.array([function_derivative_dx(x), function_derivative_dy(y)])

    'Gradient Descent on 2x² - 4x y + y⁴ + 2'

    initial_x, initial_y = -5.0, 2.5
    state = np.array([initial_x, initial_y])
    mn = gradient_descent(function_derivative, state)
    print(f'The minimum found at state = {mn}')
    # The minimum z exists at (x,y) = [-2.00730307  1.20259678]

def trial2():
    def f(x, y):
        return 2 * x ** 2 + 4 * x * y + y ** 4 + 2       # 2x² - 4x y + y⁴ + 2

    def function_derivative_dx(x, y):
        return 4 * (x-y)

    def function_derivative_dy(x, y):
        return 4 * (y **3 - x)

    def function_derivative(states):
        x, y = states[0], states[1]
        return np.array([function_derivative_dx(x, y), function_derivative_dy(x, y)])

    'Gradient Descent on 2x² - 4x y + y⁴ + 2'

    initial_x, initial_y = 2.5, 1.9
    state = np.array([initial_x, initial_y])
    mn = gradient_descent(function_derivative, state)
    print(f'The minimum found at state = {mn}')
    # The minimum z exists at (x,y) = [1.11270719 1.04406617]

if __name__ == '__main__':
    # trial1()
    trial2()