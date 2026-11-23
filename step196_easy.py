
import numpy as np

# Build all 36 possible two-dice outcomes
die = np.arange(1, 7)
d1, d2 = np.meshgrid(die, die)
omega = np.column_stack((d1.ravel(), d2.ravel()))

# Probability function
def prob(event):
    return np.mean(event(omega))

# Three probabilities
print("P(sum = 7):",
      f"{prob(lambda x: x.sum(axis=1) == 7):.4f}")

print("P(at least one 6):",
      f"{prob(lambda x: np.any(x == 6, axis=1)):.4f}")

print("P(sum = 7 or sum = 11):",
      f"{prob(lambda x: np.isin(x.sum(axis=1), [7, 11])):.4f}")

# Verify probability distribution
print("\nProbability distribution:")

total = 0

for k in range(2, 13):
    pk = prob(lambda x: x.sum(axis=1) == k)
    total += pk
    print(f"P(sum = {k:2d}): {pk:.4f}")

print("\nTotal probability:", f"{total:.4f}")
print("Axiom verified:", np.isclose(total, 1.0))
