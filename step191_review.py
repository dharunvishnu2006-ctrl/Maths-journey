import math
import numpy as np

# UNDERSTAND:
# - Input: n (int), b (int), rng (Generator) -> Output: list of 1D index arrays for 1 shuffled epoch.
# - Rules: all 0..n-1 appear once; batch count = ceil(n / b); remainder handled safely.
# - Hand-worked cases:
#   - (n=10, b=3): 4 batches, last size = 1
#   - (n=8,  b=4): 2 batches, last size = 4
#   - (n=5, b=10): 1 batch,  last size = 5
# - Pattern: Shuffled index partitioning.

# PLAN:
# 1. Define make_batches(n, b, rng) using rng.permutation(n) and slice steps of b.
# 2. Add a test helper checking uniqueness (0..n-1), batch count, and last batch size.
# 3. Test edge cases first: (10, 3), (8, 4), and (5, 10).


# 1. Define make_batches(n, b, rng) using rng.permutation(n) and slice steps of b.
def make_batches(n, b, rng):
    perm = rng.permutation(n)
    return [perm[start:start + b] for start in range(0, n, b)]


# 2. Add a test helper checking uniqueness (0..n-1), batch count, and last batch size.
def check_batches(n, b, expected_count, expected_last_size, rng):
    batches = make_batches(n, b, rng)
    flat = np.concatenate(batches)
    assert sorted(flat.tolist()) == list(
        range(n)
    ), f"Missing/duplicate indices for n={n}, b={b}"
    assert (
        len(batches) == math.ceil(n / b) == expected_count
    ), f"Expected {expected_count} batches, got {len(batches)}"
    assert (
        len(batches[-1]) == expected_last_size
    ), f"Expected last size {expected_last_size}, got {len(batches[-1])}"
    print(
        f"PASS (n={n}, b={b}): batches={len(batches)},"
        f" last_size={len(batches[-1])}, first_batch={batches[0]}"
    )


# 3. Test edge cases first: (10, 3), (8, 4), and (5, 10).
rng = np.random.default_rng(42)
check_batches(10, 3, expected_count=4, expected_last_size=1, rng=rng)
check_batches(8, 4, expected_count=2, expected_last_size=4, rng=rng)
check_batches(5, 10, expected_count=1, expected_last_size=5, rng=rng)