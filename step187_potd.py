def can_jump(nums):
    # 1. Track farthest reachable index
    farthest = 0
    # 2. Walk through array
    for i, jump in enumerate(nums):
        # 3. Stuck check
        if i > farthest:
            return False
        # 4. Update farthest
        farthest = max(farthest, i + jump)
        # 5. Early exit if end reached
        if farthest >= len(nums) - 1:
            return True
    # 6. Default return
    return True

# Tests
tests = [[0], [2, 0, 1], [3, 2, 1, 0, 4], [2, 3, 1, 1, 4]]
for t in tests:
    print(f"{t} -> {can_jump(t)}")