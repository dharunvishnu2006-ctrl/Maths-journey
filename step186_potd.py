import heapq

def findKthLargest(nums, k):
    # 1. Initialize an empty min-heap
    heap = []

    # 2. Iterate through each number in nums:
    for num in nums:
        # - Push the number into the min-heap
        heapq.heappush(heap, num)
        # - If heap size exceeds k, pop the smallest element
        if len(heap) > k:
            heapq.heappop(heap)

    # 3. The heap holds the k largest numbers; root (heap[0]) is the k-th largest
    # 4. Return heap[0]
    return heap[0]


# --- Test Cases ---

test_cases = [
    ([3, 2, 1, 5, 6, 4], 2),
    ([3, 2, 3, 1, 2, 4, 5, 5, 6], 4),
    ([7], 1),
    ([5, 5, 5], 2),
]

for i, (nums, k) in enumerate(test_cases, 1):
    result = findKthLargest(nums, k)
    print(f"Test {i}: nums={nums}, k={k} -> {result}")