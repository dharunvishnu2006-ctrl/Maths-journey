# UNDERSTAND:
# - Input: list `temps` of daily temperatures (length n).
# - Output: list `ans` of length n where ans[i] = number of days after i until a strictly warmer temp (temps[j] > temps[i], j > i), or 0 if none exists.
# - Edge cases: empty list [], single element [50], strictly decreasing [90, 80, 70], all equal [70, 70, 70], strictly increasing [30, 40, 50, 60].

# PLAN:
# 1. Initialize ans = [0] * len(temps) and an empty monotonic decreasing stack storing indices of unresolved days.
# 2. Loop through (i, temp) in enumerate(temps):
#    - While stack is non-empty and temp > temps[stack[-1]]:
#        pop prev_i = stack.pop() and set ans[prev_i] = i - prev_i.
#    - Push current index i onto stack.
# 3. Return ans (any indices left in stack keep their default 0).

def daily_temperatures(temps):
    # 1. Initialize ans and monotonic decreasing stack of indices.
    ans = [0] * len(temps)
    stack = []

    # 2. Resolve warmer days using the stack.
    for i, temp in enumerate(temps):
        while stack and temp > temps[stack[-1]]:
            prev_i = stack.pop()
            ans[prev_i] = i - prev_i
        stack.append(i)

    # 3. Return result.
    return ans

# TEST (Edge cases first):
assert daily_temperatures([]) == []
assert daily_temperatures([50]) == [0]
assert daily_temperatures([90, 80, 70]) == [0, 0, 0]
assert daily_temperatures([70, 70, 70]) == [0, 0, 0]

# Standard examples:
assert daily_temperatures([73, 74, 75, 71, 69, 72, 76, 73]) == [1, 1, 4, 2, 1, 1, 0, 0]
assert daily_temperatures([30, 40, 50, 60]) == [1, 1, 1, 0]
assert daily_temperatures([30, 60, 90]) == [1, 1, 0]

print("Edge cases PASSED:", daily_temperatures([]), daily_temperatures([50]), daily_temperatures([90, 80, 70]), daily_temperatures([70, 70, 70]))
print("Examples PASSED:", daily_temperatures([73, 74, 75, 71, 69, 72, 76, 73]))