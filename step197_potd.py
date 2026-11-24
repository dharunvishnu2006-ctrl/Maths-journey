def can_jump(nums):
    farthest = 0

    for i in range(len(nums)):
        if i > farthest:
            return False

        farthest = max(farthest, i + nums[i])

    return True


print(can_jump([2, 3, 1, 1, 4]))
print(can_jump([3, 2, 1, 0, 4]))
print(can_jump([0]))
print(can_jump([0, 1]))
print(can_jump([2, 0, 0]))