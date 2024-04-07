from colors import bcolors
import math


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

    print("{:<10} {:<15} {:<15} {:<15}".format("Iteration", "x0", "x1", "y"))  # Print table header

    # Check if the initial guesses are too close
    if abs(f(x0)-f(x1)) <= TOL:
        print("Method cannot continue: values x0 and x1 are too close to each other.")
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


if __name__ == '__main__':
    f = lambda x: x**2 - 5*x + 2  # Define the function
    x0 = 0.0  # Initial guess for the root
    x1 = 1.0  # Initial guess for the root

    # Call Secant method to find the root
    roots = secant_method(f, x0, x1)

    if roots is not None:
        # Print the approximate root found by the method
        print(bcolors.OKBLUE, f"\nThe equation f(x) has an approximate root at x = {roots}", bcolors.ENDC)
