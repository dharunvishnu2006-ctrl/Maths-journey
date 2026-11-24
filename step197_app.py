import numpy as np

def total_probability(shares, rates):

    if len(shares) != len(rates):
        raise ValueError("List lengths must match")

    if len(shares) == 0:
        raise ValueError("Lists cannot be empty")

    if any(s < 0 or s > 1 for s in shares):
        raise ValueError("Invalid share")

    if any(r < 0 or r > 1 for r in rates):
        raise ValueError("Invalid failure rate")

    if not np.isclose(sum(shares), 1):
        raise ValueError("Shares must sum to 1")

    return sum(s * r for s, r in zip(shares, rates))


# Regional data
shares = [0.5, 0.3, 0.2]
rates = [0.02, 0.05, 0.10]

# Calculate probabilities
p_fail = total_probability(shares, rates)
p_success = 1 - p_fail

print("P(fail) =", p_fail)
print("P(success) =", p_success)

# 6a. Assign each job a region using regional shares.
rng = np.random.default_rng(42)
n = 100_000

regions = rng.choice(3, size=n, p=shares)

# 6b. Apply each region's failure rate.
job_rates = np.array(rates)[regions]
failures = rng.random(n) < job_rates

# 6c. Calculate simulated failure probability.
p_simulated = np.mean(failures)

# 7. Compare theoretical and simulated results.
print("Simulated P(fail) =", round(p_simulated, 4))
print("Theoretical P(fail) =", round(p_fail, 4))

try:
    total_probability([0.5, 0.3], [0.02, 0.05])
except ValueError as e:
    print("Caught:", e)

try:
    total_probability([0.5, 0.5], [0.02])
except ValueError as e:
    print("Caught:", e)