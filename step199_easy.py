import numpy as np

def independent_from_counts(n_a, n_b, n_ab, total):
    p_ab = n_ab / total
    p_product = (n_a / total) * (n_b / total)

    print("P(A&B) =", p_ab)
    print("P(A)*P(B) =", p_product)

    # Exact counts: use np.isclose
    return np.isclose(p_ab, p_product)


print("Pair 1:", independent_from_counts(200, 150, 30, 1000))
print("Pair 2:", independent_from_counts(250, 150, 75, 1000))    