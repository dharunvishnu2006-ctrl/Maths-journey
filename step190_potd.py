# UNDERSTAND: area = (right - left) * min(height[left], height[right])
# PLAN: Two pointers at edges; advance the shorter one inward.
def max_area(height: list[int]) -> int:
  left = 0
  right = len(height) - 1
  best = 0

  while left < right:
    width = right - left
    # Bottle-necked strictly by the shorter of the two heights
    if height[left] < height[right]:
      area = width * height[left]
      left += 1
    else:
      area = width * height[right]
      right -= 1

    if area > best:
      best = area

  return best


if __name__ == "__main__":
  test_cases = [
      ([1, 1], 1),  # Edge: n = 2
      ([0, 2], 0),  # Edge: zero height
      ([1, 2, 3, 4], 4),  # Edge: strictly increasing
      ([4, 1, 1, 4], 12),  # Edge: equal endpoints
      ([1, 8, 6, 2, 5, 4, 8, 3, 7], 49),  # Example 1
  ]

  for h, expected in test_cases:
    res = max_area(h)
    print(f"height={h} -> res={res} | expected={expected}")
    assert res == expected