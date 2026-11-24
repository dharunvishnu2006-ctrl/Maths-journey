import numpy as np

def total_probability(shares, rates):
    total = 0

    for s in shares:
        total += s

    if not np.isclose(total, 1):
        raise ValueError("Shares must sum to 1")

    if len(shares) != len(rates):
        raise ValueError("List lengths must match")

    return sum(s * r for s, r in zip(shares, rates))


shares = [0.7, 0.2, 0.1]
rates = [0.01, 0.04, 0.08]

p_fail = total_probability(shares, rates)

print("P(fail) =", round(p_fail, 4))