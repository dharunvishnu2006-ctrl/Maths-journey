import sympy as sp

x = sp.Symbol('x')

def my_derivative(x_val):
    # derivative of (3x + 2)^2, worked out by hand
    return 2 * (3 * x_val + 2)*3

exact = float(sp.diff((3*x + 2)**2, x).subs(x, 1))

print("mine :", my_derivative(1))
print("sympy:", exact)