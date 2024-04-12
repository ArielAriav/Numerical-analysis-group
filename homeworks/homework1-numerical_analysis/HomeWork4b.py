import math

def trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    result = (f(a) + f(b)) / 2.0
    for i in range(1, n):
        result += f(a + i * h)
    result *= h
    return result

def function(x):
    return math.sin(x)

# חישוב האינטגרל בשיטת הטרפז עם 4 מקטעים
integral_approximation = trapezoidal_rule(function, 0, math.pi, 4)

# חישוב הערך האמיתי של האינטגרל
true_value = 2.0

# חישוב השגיאה
error = abs(true_value - integral_approximation)

print("Approximation of the integral using trapezoidal rule:", integral_approximation)
print("True value of the integral:", true_value)
print("Error:", error)
