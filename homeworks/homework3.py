
import math
import autograd.numpy as np
from autograd import grad


def secant_method(f, x0, x1, TOL=1e-6, N=50):
    """
    Secant method for finding approximate roots of a function.

    :param f: The original function.
    :param x0: The first initial guess for the root.
    :param x1: The second initial guess for the root.
    :param TOL: The tolerance for the root (default is 1e-6).
    :param N: The maximum number of iterations (default is 50).
    :return: The approximate root found by the method.
    """

    print("{:<10} {:<15} {:<15} {:<15}".format("Iteration", "x0", "x1", "p"))  # Print table header

    # Check if the initial guesses are too close
    if abs(f(x0)-f(x1)) <= TOL:
        print("The values x0 and x1 are too close to each other.")
        return None

    prev_x1 = x1  # Store the previous x1 value
    for i in range(N):  # Loop for maximum of N iterations
        if abs(x1 - x0) < TOL or math.isclose(abs(f(x1)), 0):  # Check the tolerance
            return x1  # Return the approximate root

        if f(x1) - f(x0) == 0:  # Check for division by zero
            print("Method cannot continue: Division by zero.")
            return None

        # Calculate next approximation using Secant method formula
        y = x0 - f(x0) * ((x1 - x0) / (f(x1) - f(x0)))

        print("{:<10} {:<15.9f} {:<15.9f} {:<15.9f}".format(i, x0, x1, y))  # Print iteration details

        # Check for non-convergence (if consecutive approximations are not getting closer)
        if abs(y - prev_x1) < TOL:
            print("Method may not converge: Consecutive approximations are not getting closer.")
            return None

        prev_x1 = x1  # Update the previous x1 value
        x0 = x1  # Update x0 for next iteration
        x1 = y  # Update x1 for next iteration

    return x1  # Return the last approximation if maximum iterations reached


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


if __name__ == '__main__':
    x0 = 4.0  # Initial guess for the root
    x1 = 5.0  # Initial guess for the root
    df = grad(f)  # Compute the derivative of the function using Autograd

    # Call Secant method to find the root
    roots = secant_method(f, x0, x1)

    # Call Newton Raphson method to find the root
    roots1 = newton_raphson(f, df, x0)

    if roots is not None:
        # Print the approximate root found by the secant method
        print( f"\nThe equation f(x) has an approximate root according to secant method at x = {roots}")

    if roots1 is not None:
        # Print the approximate root found by the newton raphson method
        print(f"\nThe equation f(x) has an approximate root according to newton raphson method at x = {roots}")