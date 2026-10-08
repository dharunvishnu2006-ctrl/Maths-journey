def longest_consecutive(nums: list[int]) -> int:
    if not nums:
        return 0

    num_set = set(nums)
    max_streak = 0

    for num in num_set:
        if num - 1 not in num_set:
            current_num = num
            current_streak = 1

            while current_num + 1 in num_set:
                current_num += 1
                current_streak += 1

            max_streak = max(max_streak, current_streak)

    return max_streak

print("Edge 1 (empty):", longest_consecutive([]))
print("Edge 2 (duplicates):", longest_consecutive([1, 2, 0, 1]))
print("Edge 3 (negatives):", longest_consecutive([-2, -1, 0, 1]))

print("Example 1:", longest_consecutive([100, 4, 200, 1, 3, 2]))
print("Example 2:", longest_consecutive([0, 3, 7, 2, 5, 8, 4, 6, 0, 1]))