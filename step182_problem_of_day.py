from typing import List

def dailyTemperatures(temperatures: List[int]) -> List[int]:
    # 1. ans = [0] * len(temperatures), stack = []
    ans = [0] * len(temperatures)
    stack = []
    
    # 2. For each index i, temp t in enumerate(temperatures):
    for i, t in enumerate(temperatures):
        # 3. while stack and t > temperatures[stack[-1]]:
        while stack and t > temperatures[stack[-1]]:
            # 4. prev = stack.pop(); ans[prev] = i - prev
            prev = stack.pop()
            ans[prev] = i - prev
        # 5. stack.append(i)
        stack.append(i)
        
    # 6. Return ans
    return ans

# Edge cases first
tests = [
    [80, 70, 60],
    [70, 70, 70],
    [50],
    [30, 40, 50],
    [73, 74, 75, 71, 69, 72, 76, 73]
]

for t in tests:
    print(f"{t} -> {dailyTemperatures(t)}")