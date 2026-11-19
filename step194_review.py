import math
import numpy as np


# UNDERSTAND
# Compute MSE, MAE, Huber(delta), and R^2 for 1D arrays y and y_hat, returning a dict of floats.
# Handle constant y (SS_tot == 0) by returning float("nan") for R^2 instead of dividing by zero.
# Test 4 holds because when delta -> 0 (|e| > delta), delta * (|e| - 0.5 * delta) ≈ delta * |e|, so Huber / delta -> MAE.

# PLAN
# 1. Convert y and y_hat to float arrays; compute e = y - y_hat and abs_e = |e|.
# 2. Compute mse = mean(e^2) and mae = mean(abs_e).
# 3. Compute Huber piecewise (0.5 * e^2 if abs_e <= delta else delta * (abs_e - 0.5 * delta)) and take mean.
# 4. Compute SS_res = sum(e^2) and SS_tot = sum((y - mean(y))^2); r2 = nan if SS_tot == 0 else 1 - SS_res / SS_tot.
# 5. Return {"mse": float(mse), "mae": float(mae), "huber": float(huber), "r2": float(r2)}.


def regression_report(y, y_hat, delta):
    # 1. Convert y and y_hat to float arrays; compute e = y - y_hat and abs_e = |e|.
    y_arr = np.asarray(y, dtype=float)
    y_hat_arr = np.asarray(y_hat, dtype=float)
    e = y_arr - y_hat_arr
    abs_e = np.abs(e)

    # 2. Compute mse = mean(e^2) and mae = mean(abs_e).
    mse = np.mean(e ** 2)
    mae = np.mean(abs_e)

    # 3. Compute Huber piecewise (0.5 * e^2 if abs_e <= delta else delta * (abs_e - 0.5 * delta)) and take mean.
    huber_terms = np.where(
        abs_e <= delta,
        0.5 * (e ** 2),
        delta * (abs_e - 0.5 * delta),
    )
    huber = np.mean(huber_terms)

    # 4. Compute SS_res = sum(e^2) and SS_tot = sum((y - mean(y))^2); r2 = nan if SS_tot == 0 else 1 - SS_res / SS_tot.
    ss_res = np.sum(e ** 2)
    ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
    r2 = float("nan") if ss_tot == 0 else 1.0 - (ss_res / ss_tot)

    # 5. Return {"mse": float(mse), "mae": float(mae), "huber": float(huber), "r2": float(r2)}.
    return {
        "mse": float(mse),
        "mae": float(mae),
        "huber": float(huber),
        "r2": float(r2),
    }


if __name__ == "__main__":
    # Test 1: Perfect predictions
    r1 = regression_report([1, 2, 3, 4], [1, 2, 3, 4], delta=1.0)
    assert r1 == {"mse": 0.0, "mae": 0.0, "huber": 0.0, "r2": 1.0}
    print("Test 1 (perfect):", r1)

    # Test 2: Constant y -> R^2 is nan, no crash
    r2 = regression_report([7, 7, 7], [6, 7, 8], delta=1.0)
    assert math.isnan(r2["r2"])
    print("Test 2 (constant y):", r2)

    # Test 3: Huge delta (1e6) -> Huber == 0.5 * MSE
    r3 = regression_report([5, 7, 9, 30], [6, 6, 10, 10], delta=1e6)
    assert math.isclose(r3["huber"], 0.5 * r3["mse"])
    print("Test 3 (huge delta):", r3)

    # Test 4: Tiny delta (1e-6) -> Huber / delta ≈ MAE
    r4 = regression_report([5, 7, 9, 30], [6, 6, 10, 10], delta=1e-6)
    assert math.isclose(r4["huber"] / 1e-6, r4["mae"], rel_tol=1e-5)
    print("Test 4 (tiny delta):", r4, "Huber/delta =", r4["huber"] / 1e-6)

    # Test 5: Challenge 1 numbers
    r5 = regression_report([5, 7, 9, 30], [6, 6, 10, 10], delta=3.0)
    assert math.isclose(r5["mse"], 100.75)
    assert math.isclose(r5["mae"], 5.75)
    assert math.isclose(r5["huber"], 14.25)
    assert math.isclose(r5["r2"], 1.75 / 404.75)
    print("Test 5 (Challenge 1):", r5)