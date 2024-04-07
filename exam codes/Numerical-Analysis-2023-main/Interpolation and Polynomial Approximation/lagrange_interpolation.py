from colors import bcolors


def lagrange_interpolation(x_data, y_data, x):
    """
    Lagrange Interpolation

    Parameters:
    x_data (list): List of x-values for data points.
    y_data (list): List of y-values for data points.
    x (float): The x-value where you want to evaluate the interpolated polynomial.

    Returns:
    float: The interpolated y-value at the given x.

    Raises:
    ValueError: If the lengths of x_data and y_data are not equal or if they are empty.
    """
    # Input validation
    if not x_data or not y_data or len(x_data) != len(y_data):
        raise ValueError("x_data and y_data must be non-empty lists of equal lengths.")

    n = len(x_data)  # Number of data points
    result = 0.0

    for i in range(n):
        term = y_data[i]  # Initialize the term with the y-value at the i-th data point
        for j in range(n):
            if i != j:  # Exclude the current data point when calculating the term
                term *= (x - x_data[j]) / (x_data[i] - x_data[j])  # Calculate the term using Lagrange basis polynomials
        result += term  # Add the term to the result

    return result


if __name__ == '__main__':
    try:
        # Example data points
        x_data = [1, 2, 5]
        y_data = [1, 0, 2]
        x_interpolate = 3  # The x-value where you want to interpolate

        # Interpolate y-value at x_interpolate
        y_interpolate = lagrange_interpolation(x_data, y_data, x_interpolate)

        # Print the interpolated value
        print(bcolors.OKBLUE, "\nInterpolated value at x =", x_interpolate, "is y =", y_interpolate, bcolors.ENDC)

    except ValueError as ve:
        print(bcolors.FAIL, "Error:", ve, bcolors.ENDC)
