
# Part 1: Import libraries
import numpy as np
from scipy.stats import geom

# Part 2: Set probability
p = 0.3

# Part 3: Calculate theoretical probabilities
first_on_3 = geom.pmf(3, p)
within_3 = geom.cdf(3, p)

print("First answer on call 3:", first_on_3)
print("Answer within 3 calls:", within_3)

# Part 4: Simulate 10,000 agents
calls = geom.rvs(p, size=10000, random_state=42)
fraction = np.mean(calls <= 3)

print("Simulated fraction:", fraction)

# Part 5: Verify PMF sum
k = np.arange(1, 101)
pmf_sum = np.sum(geom.pmf(k, p))

print("PMF sum:", pmf_sum)

# Part 6: Remaining tail probability
print("Remaining tail probability:", 0.7**100)

# Part 7: Test edge cases
for p in [1, 0]:
    print("\np =", p)
    print("PMF:", geom.pmf(1, p))
    print("CDF:", geom.cdf(3, p))
