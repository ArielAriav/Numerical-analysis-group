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
    :param TOL: The tolerance for the root (default is 1e-6).
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

        # Print iteration number, current approximation p0, and next approximation p
        print("{:<10} {:<15.9f} {:<15.9f} ".format(i, x0, x1))
        x0 = x1  # Update x0 for next iteration

    # Inform the user if convergence was not achieved
    print("Method did not converge within the specified number of iterations.")
    return None


# Define the function
def f(x):
    return x**3 - 3*x**2


# ______________________________________________________________________________________________________
if __name__ == '__main__':

    x0 = 5.0  # Initial guess for the root (as a float)
    df = grad(f)  # Compute the derivative of the function using Autograd

    # Call Newton Raphson method to find the root
    roots = newton_raphson(f, df, x0)

    if roots is not None:
        # Print the approximate root found by the method
        print(bcolors.OKBLUE, f"\nThe equation f(x) has an approximate root at x = {roots}", bcolors.ENDC)
