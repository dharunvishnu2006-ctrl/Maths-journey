def rob(nums):
    prev2 = 0
    prev1 = 0

    for money in nums:
        current = max(money + prev2, prev1)
        prev2 = prev1
        prev1 = current

    return prev1


print(rob([1, 2, 3, 1]))
print(rob([2, 7, 9, 3, 1]))
print(rob([5]))
print(rob([0, 0, 0]))
