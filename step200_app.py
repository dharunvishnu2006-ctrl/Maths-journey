from collections import Counter
from scipy.stats import poisson

counts = [3, 2, 4, 1, 5, 2, 3, 0, 2, 4,
          3, 6, 1, 2, 3, 4, 2, 5, 3, 1,
          7, 2, 3, 4, 0, 1, 2, 3, 5, 4,
          2, 3, 6, 1, 4, 2, 3, 5, 1, 4]

# 1. Check empty list
if not counts:
    raise ValueError("Counts list is empty")

# 2. Calculate lambda
lam = sum(counts) / len(counts)
print("Lambda:", lam)

# 3. Count actual frequencies
actual = Counter(counts)

# 4 and 6. Calculate and print expected frequencies
print("k | Actual | Expected")
for k in range(8):
    expected = len(counts) * poisson.pmf(k, mu=lam)
    print(k, actual.get(k, 0), round(expected, 2))