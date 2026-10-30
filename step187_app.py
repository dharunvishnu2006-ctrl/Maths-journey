import sympy as sp

# 1. SymPy gradient at (2, 1)
w1, w2 = sp.symbols('w1 w2')
L = (2*w1 + w2 - 5)**2

gw1 = sp.diff(L, w1).subs(w1, 2).subs(w2, 1)
gw2 = sp.diff(L, w2).subs(w1, 2).subs(w2, 1)
print("SymPy at (2, 1):", [gw1, gw2])

# 2. Central difference at (2, 1)
def loss(a, b):
    return (2*a + b - 5)**2

h = 1e-5
num1 = (loss(2 + h, 1) - loss(2 - h, 1)) / (2*h)
num2 = (loss(2, 1 + h) - loss(2, 1 - h)) / (2*h)
print("Numeric at (2, 1):", [round(num1, 4), round(num2, 4)])