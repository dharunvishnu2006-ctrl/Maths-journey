# UNDERSTAND: Positive ints -> monotonic window sum. Find shortest subarray >= target, else 0.
# Edge cases: [], impossible target -> 0; single hit -> 1; full array -> len(nums).
# PLAN:
# 1. Init left = 0, window_sum = 0, min_len = inf.
# 2. Expand right, add nums[right] to window_sum.
# 3. While window_sum >= target, update min_len, subtract nums[left], increment left.
# 4. Return min_len (or 0) and test edge cases first.

def min_subarray_len(target: int, nums: list[int]) -> int:
    left, window_sum, min_len = 0, 0, float("inf")
    for right, val in enumerate(nums):
        window_sum += val
        while window_sum >= target:
            min_len = min(min_len, right - left + 1)
            window_sum -= nums[left]
            left += 1
    return min_len if min_len != float("inf") else 0

tests = [(5, [], 0), (11, [1]*5, 0), (4, [1, 4, 4], 1), (15, [1, 2, 3, 4, 5], 5), (7, [2, 3, 1, 2, 4, 3], 2)]
for t, arr, exp in tests:
    got = min_subarray_len(t, arr)
    assert got == exp
    print(f"PASS: target={t}, nums={arr} -> {got}")