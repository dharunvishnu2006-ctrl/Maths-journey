import numpy as np


def binary_report(y, p, gamma=2.0):
    # 1. Convert y and p to numpy arrays (float) and validate matching shapes / non-empty input.
    y_arr = np.asarray(y, dtype=np.float64)
    p_arr = np.asarray(p, dtype=np.float64)
    if y_arr.shape != p_arr.shape or y_arr.size == 0:
        raise ValueError("y and p must be non-empty and have the same shape.")

    # 2. Clip p to [1e-15, 1 - 1e-15] for numerical stability in log().
    eps = 1e-15
    p_clipped = np.clip(p_arr, eps, 1.0 - eps)

    # 3. Compute p_t = np.where(y == 1, p, 1.0 - p).
    p_t = np.where(y_arr == 1.0, p_clipped, 1.0 - p_clipped)

    # 4. Compute per-sample bce = -np.log(p_t) and focal = ((1.0 - p_t) ** gamma) * bce.
    bce = -np.log(p_t)
    focal = ((1.0 - p_t) ** gamma) * bce

    # 5. Compute accuracy = np.mean((p >= 0.5) == (y == 1)) using unclipped p.
    preds = (p_arr >= 0.5).astype(np.float64)
    accuracy = np.mean(preds == y_arr)

    # 6. Return {"bce": float(np.mean(bce)), "focal": float(np.mean(focal)), "accuracy": float(accuracy)}.
    return {
        "bce": float(np.mean(bce)),
        "focal": float(np.mean(focal)),
        "accuracy": float(accuracy),
    }


# 7. Write 4 isolated test functions (edge cases first) and run them.
def test_perfect_confident():
    res = binary_report([1, 0], [1.0, 0.0], gamma=2.0)
    assert np.isfinite(res["bce"]) and 0.0 < res["bce"] < 1e-12
    assert np.isfinite(res["focal"]) and 0.0 <= res["focal"] < 1e-40
    assert res["accuracy"] == 1.0


def test_gamma_zero():
    y = [1, 0, 1, 0]
    p = [0.85, 0.2, 0.4, 0.9]
    res = binary_report(y, p, gamma=0.0)
    assert np.isclose(res["focal"], res["bce"], atol=1e-12)


def test_flip_symmetry():
    y = np.array([1, 0, 1, 0, 1])
    p = np.array([0.8, 0.3, 0.1, 0.4, 0.95])
    res_orig = binary_report(y, p, gamma=2.0)
    res_flip = binary_report(1 - y, 1 - p, gamma=2.0)
    assert np.isclose(res_orig["bce"], res_flip["bce"], atol=1e-12)
    assert np.isclose(res_orig["focal"], res_flip["focal"], atol=1e-12)


def test_challenge_1():
    res = binary_report([1, 0, 1], [0.8, 0.3, 0.1], gamma=2.0)
    assert np.isclose(res["bce"], 0.9608, atol=1e-4)
    assert np.isclose(res["focal"], 0.6354, atol=1e-4)
    assert np.isclose(res["accuracy"], 2.0 / 3.0, atol=1e-4)


if __name__ == "__main__":
    test_perfect_confident()
    test_gamma_zero()
    test_flip_symmetry()
    test_challenge_1()
    print("All 4 tests passed!")