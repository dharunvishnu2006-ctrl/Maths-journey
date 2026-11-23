import heapq

# Understand:
# Find k points closest to origin (0, 0).
# Use squared Euclidean distance: x*x + y*y.

# Plan:
# 1. Create an empty max-heap using negative distances.
# 2. Push each point into the heap.
# 3. If heap size exceeds k, remove the farthest point.
# 4. Return the remaining k points.

def k_closest(points, k):
    heap = []

    for x, y in points:
        dist = x*x + y*y
        heapq.heappush(heap, (-dist, x, y))

        if len(heap) > k:
            heapq.heappop(heap)

    return [[x, y] for dist, x, y in heap]


# Edge case 1: Single point
def test_one_point():
    assert k_closest([[1, 1]], 1) == [[1, 1]]
    print("Single point: PASS")


# Edge case 2: Origin
def test_origin():
    assert k_closest([[0, 0], [2, 2]], 1) == [[0, 0]]
    print("Origin: PASS")


# Example tests
def test_examples():
    assert k_closest([[1, 3], [-2, 2]], 1) == [[-2, 2]]

    result = k_closest([[3, 3], [5, -1], [-2, 4]], 2)
    assert sorted(map(tuple, result)) == [(-2, 4), (3, 3)]
    print("Examples: PASS")


# Tie test
def test_ties():
    result = k_closest([[1, 0], [0, 1], [-1, 0]], 2)
    assert len(result) == 2
    assert all(x*x + y*y == 1 for x, y in result)
    print("Ties: PASS")


# k equals number of points
def test_all_points():
    points = [[1, 2], [3, 4]]
    result = k_closest(points, 2)
    assert sorted(map(tuple, result)) == sorted(map(tuple, points))
    print("All points: PASS")


# Run tests
test_one_point()
test_origin()
test_examples()
test_ties()
test_all_points()

print("\nAll tests passed!")
