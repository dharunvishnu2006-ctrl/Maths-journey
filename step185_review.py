import sympy as sp

# 1. Define Python function f(x) returning x**2 + 4*x
def f(x):
    return x**2 + 4 * x

# 2. Define numeric_derivative(f, x, h) returning (f(x + h) - f(x)) / h
def numeric_derivative(f, x, h):
    return (f(x + h) - f(x)) / h

# 3. Use SymPy to compute the exact derivative:
#    - Define symbol x
#    - Differentiate x**2 + 4*x using sp.diff
#    - Substitute x = 1 using .subs() to get exact numerical value
x_sym = sp.Symbol('x')
f_sym = x_sym**2 + 4 * x_sym
df_sym = sp.diff(f_sym, x_sym)
exact_val = float(df_sym.subs(x_sym, 1))

print(f"Exact derivative at x = 1: {exact_val:.4f}\n")

# 4. Initialize variable or flag to track the first h with error < 0.005
h_values = [1, 0.1, 0.01, 0.001]
first_h_below_threshold = None

# 5. Loop over h in [1, 0.1, 0.01, 0.001]:
for h in h_values:
    # - Calculate numeric slope using numeric_derivative
    numeric_slope = numeric_derivative(f, 1, h)
    
    # - Calculate absolute error: abs(exact - numeric)
    error = abs(exact_val - numeric_slope)
    
    # - Print h, numeric slope, and error
    print(f"h = {h:<6} | Numeric: {numeric_slope:.6f} | Error: {error:.6f}")
    
    # - If error < 0.005 and first match not yet recorded, record this h
    if error < 0.005 and first_h_below_threshold is None:
        first_h_below_threshold = h

# 6. Print the first h that achieved error < 0.005
print(f"\nFirst h with error < 0.005: {first_h_below_threshold}")