# ==============================================================================
# Plan:
# 1. Base cases:
#    - If len(nums) == 1, return nums[0] (single house, no conflicts).
#    - Initialize prev2 = nums[0] (best at index 0) and prev1 = max(nums[0], nums[1]) (best at index 1).
# 2. Loop through nums starting from index 2 to the end:
#    - Compute current = max(prev1, nums[i] + prev2).
#    - Shift pointers: prev2 = prev1, prev1 = current.
# 3. Return prev1 (holds the maximum achievable sum up to the last house).
# Pattern: Dynamic Programming with O(1) space (Fibonacci-style rolling variables).
# ==============================================================================

def rob(nums: list[int]) -> int:
    # 1. Base cases:
    #    - If len(nums) == 1, return nums[0] (single house, no conflicts).
    if len(nums) == 1:
        return nums[0]
    
    #    - Initialize prev2 = nums[0] (best at index 0) and prev1 = max(nums[0], nums[1]) (best at index 1).
    prev2 = nums[0]
    prev1 = max(nums[0], nums[1])

    # 2. Loop through nums starting from index 2 to the end:
    for i in range(2, len(nums)):
        # - Compute current = max(prev1, nums[i] + prev2).
        current = max(prev1, nums[i] + prev2)
        # - Shift pointers: prev2 = prev1, prev1 = current.
        prev2 = prev1
        prev1 = current

    # 3. Return prev1 (holds the maximum achievable sum up to the last house).
    return prev1


# Tests: Edge cases first, then examples
tests = [
    ([5], 5, "Single house"),
    ([3, 8], 8, "Two houses"),
    ([2, 1, 1, 2], 4, "Two skips"),
    ([1, 2, 3, 1], 4, "Example 1"),
    ([2, 7, 9, 3, 1], 12, "Example 2"),
]

for nums, expected, label in tests:
    result = rob(nums)
    print(f"[{label}] nums={nums} -> got: {result} | expected: {expected} | pass: {result == expected}")