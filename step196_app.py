import numpy as np

# 1. Agent delivery data
agents = {
    "Arun": [45, 4, 1],
    "Bala": [2, 0, 0],
    "Chitra": [180, 15, 5],
    "Deepa": [8, 2, 0]
}

# 2. Probability function with zero guard
def probability(counts):
    total = sum(counts)
    if total == 0:
        return np.full(3, np.nan)
    return np.array(counts) / total

# 3. Calculate probabilities and verify sums
results = {}

for name, counts in agents.items():
    p = probability(counts)
    results[name] = p

    print(name, np.round(p, 4),
          "Sum:", np.sum(p) if not np.isnan(p).any()
          else "No data")

print("\nSum-to-1 checks:")
for name, p in results.items():
    print(name, np.isclose(p.sum(), 1)
          if not np.isnan(p).any() else "No data")

# 4. Rank agents
print("\nRanking:")
for name, p in sorted(results.items(),
                      key=lambda x: -x[1][0]):
    print(name, f"{p[0]:.2%}")

# 5. Bala Monte Carlo simulation
rng = np.random.default_rng(0)

sim = rng.random((100000, 2)) < 0.85
estimate = np.mean(np.all(sim, axis=1))
exact = 0.85 ** 2

print("\nBala simulation:")
print("Simulated:", round(estimate, 4))
print("Exact:", round(exact, 4))
print("Difference:", round(abs(estimate - exact), 4))

# 6. Recommendation
print("\nRecommendation: Chitra")
print("90% on-time rate across 200 deliveries.")

# 7. Edge-case test: zero deliveries
def test_zero_deliveries():
    counts = [0, 0, 0]
    p = probability(counts)

    assert np.isnan(p).all()
    print("Zero deliveries: PASS")

# 8. Edge-case test: only cancellations
def test_only_cancellations():
    counts = [0, 0, 5]
    p = probability(counts)

    assert np.allclose(p, [0, 0, 1])
    assert np.isclose(p.sum(), 1)

    print("Only cancellations: PASS")

# Run tests
print("\nEdge-case tests:")
test_zero_deliveries()
test_only_cancellations()
