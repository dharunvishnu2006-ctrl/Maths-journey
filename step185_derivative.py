import sympy as sp

def f(x):
    return x ** 2

def numeric_derivative(f, x, h):
    return (f(x + h) - f(x)) / h

for h in [1, 0.1, 0.01, 0.001]:
    print("h =", h, " slope =", numeric_derivative(f, 3, h))

x = sp.Symbol('x')

print("d/dx x^2   =", sp.diff(x**2, x))
print("d/dx 50    =", sp.diff(50, x))
print("d/dx e^x   =", sp.diff(sp.exp(x), x))
print("d/dx sin x =", sp.diff(sp.sin(x), x))    