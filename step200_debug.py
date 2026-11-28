import numpy as np
from scipy.stats import poisson

lam = 4

# P(X >= 6)
p_at_least_6 = 1 - poisson.cdf(5, lam)
print("P(X >= 6):", p_at_least_6)

# PMF sum
k = np.arange(0, 50)
print("PMF sum:", poisson.pmf(k, lam).sum())

# Simulate 10,000 hours
hours = poisson.rvs(lam, size=10000, random_state=7)
print("Simulated P(X >= 6):", np.mean(hours >= 6))