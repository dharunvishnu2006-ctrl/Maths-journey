import numpy as np

def simulate_posterior(prior, p_e_given_h,
                       p_e_given_not_h, n, seed):

    if n <= 0:
        raise ValueError("n must be positive")

    if not (0 <= prior <= 1 and
            0 <= p_e_given_h <= 1 and
            0 <= p_e_given_not_h <= 1):
        raise ValueError("Invalid probability")

    rng = np.random.default_rng(seed)

    # Simulate rainy or dry days
    rain = rng.random(n) < prior

    # Generate clouds using weather-specific rates
    cloud_probability = np.where(
        rain, p_e_given_h, p_e_given_not_h
    )
    clouds = rng.random(n) < cloud_probability

    # Count cloudy days
    cloudy_count = np.sum(clouds)

    if cloudy_count == 0:
        raise ValueError("No cloudy days")

    # Count rainy and cloudy days
    rainy_cloudy_count = np.sum(rain & clouds)

    return rainy_cloudy_count / cloudy_count


result = simulate_posterior(0.3, 0.8, 0.25, 1_000_000, 42)

print("Simulated P(rain | clouds):", round(result, 4))
print("Paper answer:", 0.5783)

# Test 1: n = 0
try:
    simulate_posterior(0.3, 0.8, 0.25, 0, 42)
except ValueError as e:
    print("Caught:", e)

# Test 2: No clouds possible
try:
    simulate_posterior(0.3, 0.0, 0.0, 1000, 42)
except ValueError as e:
    print("Caught:", e)