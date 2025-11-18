import numpy as np
from gradient_descent import gradient_descent

if __name__ == "__main__":
    def f(x, y, z):
        return np.sin(x) + np.cos(y) + np.sin(z)

    def fderiv_dx(x, y, z):
        return np.cos(x)

    def fderiv_dy(x, y, z):
        return -np.sin(y)

    def fderiv_dz(x, y, z):
        return np.cos(z)

    def fderiv(state):
        x, y, z = state[0], state[1], state[2]
        return np.array([fderiv_dx(x, y, z), fderiv_dy(x, y, z), fderiv_dz(x, y, z)])


    def test():
        initial_x, initial_y, initial_z = [1.0, 2.0, 3.5]
        state = np.array([initial_x, initial_y, initial_z])
        mn = gradient_descent(fderiv, state)
        print(f'The minimum found at state = {mn}')
        print(f'The minimum value of the function is f(min) = {f(mn[0], mn[1], mn[2])}')

    test()