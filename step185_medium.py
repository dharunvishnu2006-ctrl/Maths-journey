import sympy as sp

# Define symbolic variable
x = sp.Symbol('x')

# 1 & 2: Derivative of x**3
f1 = x**3
df1 = sp.diff(f1, x)
val1 = df1.subs(x, 2)

print("--- Function 1: x^3 ---")
print(f"Derivative formula: {df1}")
print(f"Value at x = 2:     {val1}")

# 3: Derivative of 5x^2 + 3x + 7 at x = 1
f2 = 5*x**2 + 3*x + 7
df2 = sp.diff(f2, x)
val2 = df2.subs(x, 1)

print("\n--- Function 2: 5x^2 + 3x + 7 ---")
print(f"Derivative formula: {df2}")
print(f"Value at x = 1:     {val2}")