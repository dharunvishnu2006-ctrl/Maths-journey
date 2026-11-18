import math


def min_eating_speed(piles: list[int], h: int) -> int:
    # Understand:
    # - Find the smallest integer eating speed k (bananas/hour) so total hours sum(ceil(p / k) for p in piles) <= h.
    # - Total hours needed is monotonically non-increasing as k increases, bounded by k in [1, max(piles)].
    #
    # Plan:
    # 1. Handle empty piles (return 0) and set binary search bounds: low = 1, high = max(piles).
    # 2. While low < high, compute mid = low + (high - low) // 2 and total hours using ceiling division (p + mid - 1) // mid.
    # 3. If hours <= h, mid is feasible -> shrink right bound (high = mid); otherwise increase speed (low = mid + 1).
    # 4. Return low once the search space converges.
    if not piles:
        return 0

    low, high = 1, max(piles)

    while low < high:
        mid = low + (high - low) // 2
        hours_needed = sum((p + mid - 1) // mid for p in piles)

        if hours_needed <= h:
            high = mid
        else:
            low = mid + 1

    return low


if __name__ == "__main__":
    # Edge cases first
    assert min_eating_speed([], 5) == 0                        # empty piles
    assert min_eating_speed([1], 1) == 1                       # single pile, minimal speed
    assert min_eating_speed([1000000000], 2) == 500000000      # single large pile
    assert min_eating_speed([1, 1, 1, 1], 4) == 1              # all 1s, tight h == len(piles)
    assert min_eating_speed([5, 5, 5], 100) == 1               # huge h allows minimum speed k = 1

    # Standard examples
    assert min_eating_speed([3, 6, 7, 11], 8) == 4
    assert min_eating_speed([30, 11, 23, 4, 20], 5) == 30
    assert min_eating_speed([30, 11, 23, 4, 20], 6) == 23

    print("Edge cases passed!")
    print("Example 1 ([3, 6, 7, 11], h=8):", min_eating_speed([3, 6, 7, 11], 8))
    print("Example 2 ([30, 11, 23, 4, 20], h=5):", min_eating_speed([30, 11, 23, 4, 20], 5))
    print("Example 3 ([30, 11, 23, 4, 20], h=6):", min_eating_speed([30, 11, 23, 4, 20], 6))