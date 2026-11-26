import numpy as np

def is_independent(n_a, n_b, n_ab, total):
    p_a = n_a / total
    p_a_given_b = n_ab / n_b
    return np.isclose(p_a_given_b, p_a)

print("Promo vs late independent?",
      is_independent(200, 150, 30, 1000))