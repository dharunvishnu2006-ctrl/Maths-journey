def f(x):
    return x ** 3

def numeric_derivative(f, x, h):
    return (f(x + h) - f(x)) / h

x = 2
h_values = [0.1, 0.01, 0.001, 0.0001]

for h in h_values:
    slope = numeric_derivative(f, x, h)
    print(f"h = {h:<7}: slope = {slope:.6f}")