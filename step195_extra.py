# step195_extra.py
import numpy as np


def confusion_counts(y, p, threshold=0.5):
    # 1. Convert y and p to 1D numpy arrays (y as int/bool, p as float) and check matching shapes.
    y_arr = np.asarray(y, dtype=np.int64)
    p_arr = np.asarray(p, dtype=np.float64)
    if y_arr.shape != p_arr.shape:
        raise ValueError("y and p must have the same shape.")

    # 2. Compute boolean predictions: pred_pos = (p_arr >= threshold) and actual positives: act_pos = (y_arr == 1).
    pred_pos = p_arr >= threshold
    act_pos = y_arr == 1

    # 3. Compute TP = int(np.sum(pred_pos & act_pos)) and FP = int(np.sum(pred_pos & ~act_pos)).
    tp = int(np.sum(pred_pos & act_pos))
    fp = int(np.sum(pred_pos & ~act_pos))

    # 4. Compute TN = int(np.sum(~pred_pos & ~act_pos)) and FN = int(np.sum(~pred_pos & act_pos)).
    tn = int(np.sum(~pred_pos & ~act_pos))
    fn = int(np.sum(~pred_pos & act_pos))

    # 5. Return {"TP": TP, "FP": FP, "TN": TN, "FN": FN}.
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


# 6. Write 4 isolated test functions (test_threshold_zero, test_threshold_above_one, test_challenge_1_counts, test_sum_equals_n) and run them.
def test_threshold_zero():
    res = confusion_counts([1, 0, 1, 0], [0.0, 0.2, 0.8, 1.0], threshold=0.0)
    assert res["TN"] == 0 and res["FN"] == 0
    assert res["TP"] == 2 and res["FP"] == 2


def test_threshold_above_one():
    res = confusion_counts([1, 0, 1, 0], [0.0, 0.2, 0.8, 1.0], threshold=1.01)
    assert res["TP"] == 0 and res["FP"] == 0
    assert res["TN"] == 2 and res["FN"] == 2


def test_challenge_1_counts():
    res = confusion_counts([1, 0, 1], [0.8, 0.3, 0.1], threshold=0.5)
    assert res == {"TP": 1, "FP": 0, "TN": 1, "FN": 1}


def test_sum_equals_n():
    y = [1, 0, 1, 1, 0, 0, 1]
    p = [0.9, 0.6, 0.4, 0.5, 0.1, 0.8, 0.2]
    for thresh in [0.0, 0.25, 0.5, 0.75, 1.01]:
        res = confusion_counts(y, p, threshold=thresh)
        assert res["TP"] + res["FP"] + res["TN"] + res["FN"] == len(y)


if __name__ == "__main__":
    test_threshold_zero()
    test_threshold_above_one()
    test_challenge_1_counts()
    test_sum_equals_n()
    print("All 4 tests passed!")