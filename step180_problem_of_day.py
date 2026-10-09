def max_area(height: list[int]) -> int:

  left = 0
  right = len(height) - 1
  max_water = 0

  while left < right:

    current_area = (right - left) * min(height[left], height[right])

    max_water = max(max_water, current_area)

    if height[left] < height[right]:
      left += 1
    else:
      right -= 1

  return max_water
edge_cases = [
    [1, 1],
    [0, 2],
    [5, 5, 5, 5],
    [5, 4, 3, 2, 1],
]
main_example = [1, 8, 6, 2, 5, 4, 8, 3, 7]

print("Edge Cases:")
for case in edge_cases:
  print(f"{case} -> {max_area(case)}")

print("\nMain Example:")
print(f"{main_example} -> {max_area(main_example)}")