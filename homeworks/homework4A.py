from matrix_utility import *


def linearInterpolation(table_points, point):
    """
    Perform linear interpolation to estimate the value at a given point based on table points.

    Parameters:
    table_points (list of tuples): List of (x, y) pairs representing table points.
    point (float): The x-value where you want to estimate the y-value.

    Returns:
    None: Prints the interpolated value.

    Raises:
    None
    """
    # Iterate through each pair of adjacent table points
    for i in range(len(table_points) - 1):
        # Extract x and y values of the current and next table points
        x1, y1 = table_points[i]
        x2, y2 = table_points[i + 1]

        # Check if the point lies within the range of the current and next table points
        if x1 <= point < x2:
            # Calculate the slope and perform linear interpolation
            m = (y1 - y2) / (x1 - x2)
            result = y1 + m * (point - x1)
            # Print the interpolated value with higher precision
            print("The approximation of the point", point, "is:", "{:.10f}".format(result))
            return
        # Handle the case where the point matches a table point exactly
        elif point == x1:
            print("The point", point, "matches a table point exactly. Interpolation is not needed.")
            return
    # If the point is outside the range of the table points, print a message
    print("The point", point, "is outside the range of the table points.")


def GaussJordanElimination(matrix, vector):
    """
    Function for solving a linear equation using Gauss-Jordan elimination method
    :param matrix: Matrix nxn
    :param vector: Vector n
    :return: Solve Ax=b -> x=A(-1)b
    """
    # Pivoting process
    matrix, vector = RowXchange(matrix, vector)
    # Inverse matrix calculation
    invert = InverseMatrix(matrix, vector)
    return MulMatrixVector(invert, vector)


def DecomposeLU(matrix, is_upper):
    """
    Function for decomposition into an upper or lower triangular matrix (U/L)
    :param matrix: Matrix nxn
    :param is_upper: Boolean indicating whether to decompose into U (True) or L (False)
    :return: U or L matrix
    """
    result = MakeIMatrix(len(matrix), len(matrix))
    for i in range(len(matrix[0])):
        matrix, _ = RowXchageZero(matrix, [])
        for j in range(i + 1, len(matrix)):
            elementary = MakeIMatrix(len(matrix[0]), len(matrix))
            sign = 1 if is_upper else -1
            elementary[j][i] = -sign * (matrix[j][i]) / matrix[i][i]
            if not is_upper:
                result[j][i] = sign * (matrix[j][i]) / matrix[i][i]
            matrix = MultiplyMatrix(elementary, matrix)
    return MultiplyMatrix(result, matrix)


def SolveUsingLU(matrix, vector):
    """
    Function for solving a linear equation by decomposing LU
    :param matrix: Matrix nxn
    :param vector: Vector n
    :return: Solve Ax=b -> x=U(-1)L(-1)b
    """
    matrixU = DecomposeLU(matrix, is_upper=True)
    matrixL = DecomposeLU(matrix, is_upper=False)
    if matrixU is None or matrixL is None:
        return None
    return MultiplyMatrix(InverseMatrix(matrixU), MultiplyMatrix(InverseMatrix(matrixL), vector))


def solveMatrix(matrixA, vectorb):
    detA = Determinant(matrixA, 1)


    if detA != 0:

        result = GaussJordanElimination(matrixA, vectorb)

        return result
    else:
        print("Singular Matrix - Unable to solve.")
        return None


def polynomialInterpolation(table_points, x):
    if len(table_points) < len(table_points[0]):
        print("Insufficient data points for interpolation.")
        return None

    matrix = [[point[0] ** i for i in range(len(table_points))] for point in table_points]
    b = [[point[1]] for point in table_points]


    matrixSol = solveMatrix(matrix, b)

    if matrixSol is not None:
        result = sum([matrixSol[i][0] * (x ** i) for i in range(len(matrixSol))])

        return result
    else:
        return None


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

    # Check if the interpolation point matches any of the given data points
    for i in range(n):
        if x == x_data[i]:
            return y_data[i]

    # If the interpolation point does not match any data point, proceed with Lagrange interpolation
    result = 0.0

    for i in range(n):
        term = y_data[i]  # Initialize the term with the y-value at the i data point
        for j in range(n):
            if i != j:  # Exclude the current data point when calculating the term
                term *= (x - x_data[j]) / (x_data[i] - x_data[j])  # Calculate the term using Lagrange basis polynomials
        result += term  # Add the term to the result

    return result

if __name__ == '__main__':
    x_data = [1, 2, 5]
    y_data = [1, 0, 2]
    table_points = [(1, 1), (2, 0), (5, 2)]
    x = 1.35
    keepRunning = True
    while keepRunning:
        print("Please choose: ")
        print("1. Polynomial interpolation")
        print("2. linear interpolation")
        print("3. lagrange interpolation")
        print("4. Exit menu")
        choice = input()
        if choice == '1':
            y = polynomialInterpolation(table_points, x)
            print("Interpolated value at x =", x, "is y =", y)
        elif choice == '2':
            linearInterpolation(table_points, x)

        elif choice == '3':
            y = lagrange_interpolation(x_data, y_data, x)
            print("Interpolated value at x =", x, "is y =", y)
        elif choice == '4':
            print("Goodbye")
            keepRunning = False
        else:
            print("Invalid input. Please try again.")
