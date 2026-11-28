def maxArea(height):
    left = 0
    right = len(height) - 1
    max_area = 0

    while left < right:
        width = right - left
        water_height = min(height[left], height[right])
        area = width * water_height

        max_area = max(max_area, area)

        if height[left] < height[right]:
            left += 1
        else:
            right -= 1

    return max_area

# Test 1
height = [1, 8, 6, 2, 5, 4, 8, 3, 7]
print(maxArea(height))

# Test 2
height = [1, 1]
print(maxArea(height))

# Test 3
height = [1, 2, 1]
print(maxArea(height))

# Test 4
height = [0, 0]
print(maxArea(height))
