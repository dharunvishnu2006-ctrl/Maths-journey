import numpy as np

# Probability simulation function
def mc_prob(event, n_coins, n_trials, seed):
    rng = np.random.default_rng(seed)

    flips = rng.integers(0, 2, size=(n_trials, n_coins))
    outcomes = event(flips)

    return np.mean(outcomes)


# TEST 1: Certain event
def test_certain():
    p = mc_prob(
        lambda x: np.ones(len(x), dtype=bool),
        3, 100000, 0
    )
    assert p == 1.0
    print("Certain event: PASS")


# TEST 2: Impossible event
def test_impossible():
    p = mc_prob(
        lambda x: x.sum(axis=1) == 4,
        3, 100000, 0
    )
    assert p == 0.0
    print("Impossible event: PASS")


# TEST 3: Exactly 2 heads
def test_exactly_two():
    p = mc_prob(
        lambda x: x.sum(axis=1) == 2,
        3, 100000, 0
    )
    assert abs(p - 0.375) < 0.01
    print("Exactly 2 heads: PASS")


# TEST 4: At least 1 head
def test_at_least_one():
    p = mc_prob(
        lambda x: x.sum(axis=1) >= 1,
        3, 100000, 0
    )
    assert abs(p - 0.875) < 0.01
    print("At least 1 head: PASS")


# Run all tests
if __name__ == "__main__":
    test_certain()
    test_impossible()
    test_exactly_two()
    test_at_least_one()

    print("\nAll 4 tests passed!")
