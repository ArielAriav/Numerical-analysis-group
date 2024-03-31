from colors import bcolors
import autograd.numpy as np
from autograd import grad
import math


def newton_raphson(f, df, x0, TOL=1e-6, N=100):
    """
    Newton Raphson method for finding approximate roots of a function.

    :param f: The original function.
    :param df: The derivative of the function.
    :param x0: The initial guess for the root.
    :param TOL: The tolerance for the root(default is 1e-6).
    :param N: The maximum number of iterations (default is 100).
    :return: The approximate root found by the method.
    """

    print("{:<10} {:<15} {:<15} ".format("Iteration", "xo", "x1"))  # Print table header
    for i in range(N):  # Loop for maximum of N iterations
        if df(x0) == 0:  # Check if derivative is zero at x0
            print("Derivative is zero at x0, method cannot continue.")
            return None

        # Calculate next approximation using Newton Raphson formula
        x1 = x0 - f(x0) / df(x0)

        if abs(x1 - x0) < TOL or math.isclose(abs(f(x1)), 0):  # Check if tolerance is reached
            return x1  # Return the approximate root
        # Print iteration number, current approximation x0, and next approximation x1
        print("{:<10} {:<15.9f} {:<15.9f} ".format(i, x0, x1))
        x0 = x1  # Update p0 for next iteration

    return x1  # Return the last approximation if maximum iterations reached


# Define the input function
def f(x):
    return x**3 - 3*x**2


# ________________________________________________________________________________________________________
if __name__ == '__main__':

    df = grad(f)  # Calculation of the derivative of f
    x0 = -5.0  # Initial guess for the root (must be floated)

    # Call Newton Raphson method to find the root
    roots = newton_raphson(f, df, x0)

    if roots is not None:
        # Print the approximate root found by the method
        print(bcolors.OKBLUE, "\nThe equation f(x) has an approximate root at x = {:<15.9f} ".format(roots),
              bcolors.ENDC,)
