def f(x):
    return x ** 3

def numeric_derivative(f, x, h):
    return (f(x + h) - f(x)) / h

print(numeric_derivative(f, 2, 0.001))