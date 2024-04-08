import numpy as np
import math


def romberg_integration(func, a, b, n):
    """
    Romberg Integration

    Parameters:
    func (function): The function to be integrated.
    a (float): The lower limit of integration.
    b (float): The upper limit of integration.
    n (int): The number of iterations (higher value leads to better accuracy).

    Returns:
    float: The approximate definite integral of the function over [a, b].
    """
    h = b - a
    R = np.zeros((n, n), dtype=float)

    R[0, 0] = 0.5 * h * (func(a) + func(b))

    for i in range(1, n):
        h /= 2
        sum_term = 0

        for k in range(1, 2 ** i, 2):
            sum_term += func(a + k * h)

        R[i, 0] = 0.5 * R[i - 1, 0] + h * sum_term

        for j in range(1, i + 1):
            R[i, j] = R[i, j - 1] + (R[i, j - 1] - R[i - 1, j - 1]) / ((4 ** j) - 1)

    return R[n - 1, n - 1]


def f(x):
    return ((6 * x ** 2) - math.cos(x ** 4 - x + 2)) / (x ** 2 + x + 2)


if __name__ == '__main__':

    a = -1.1
    b = 0.7
    n1 = 16
    integral = romberg_integration(f, a, b, n1)

    print(f"Division into n={n1} sections: ")
    print(f"Approximate integral in range [{a},{b}] is {integral}")

    n2 = 8
    integral2 = romberg_integration(f, a, b, n2)
    print(f"Division into n={n2} sections: ")
    print(f"Approximate integral in range [{a},{b}] is {integral2}")

    print(f"Final result: {integral:.5f}")