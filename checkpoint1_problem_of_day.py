
def minEatingSpeed(piles: list[int], h: int) -> int:
    # 1. Speed range is [1, max(piles)]
    # Minimum possible eating rate is 1; eating faster than max(piles) still consumes 1 hr per pile
    low = 1
    high = max(piles)
    ans = high

    # 2. Binary search over the speed range
    while low <= high:
        mid = (low + high) // 2

        # 3. For a speed mid, hours = sum of ceil(p / mid) across all piles
        # Using integer arithmetic (p + mid - 1) // mid to prevent float precision noise
        total_hours = sum((p + mid - 1) // mid for p in piles)

        # 4. If hours <= h, mid is feasible: record it and search left for a smaller speed
        if total_hours <= h:
            ans = mid
            high = mid - 1
        # 5. Otherwise, mid is too slow: search right for a faster speed
        else:
            low = mid + 1

    # 6. Return the minimum valid eating speed
    return ans


# --- VERIFICATION & TESTS ---
if __name__ == "__main__":
    # Edge case 1: h == len(piles) (each pile gets exactly 1 hr -> speed must cover max pile)
    assert minEatingSpeed([5, 10, 15], 3) == 15
    print("Edge Case 1 (h == len(piles)): Passed")

    # Edge case 2: Single pile with non-exact division requiring integer ceil
    assert minEatingSpeed([10], 3) == 4
    print("Edge Case 2 (single pile): Passed")

    # Edge case 3: Very large h relative to piles (minimum possible speed k = 1 works)
    assert minEatingSpeed([2, 4, 6], 1000) == 1
    print("Edge Case 3 (large h): Passed")

    # Edge case 4: Max constraints (piles[i] up to 10^9)
    assert minEatingSpeed([10**9], 2) == 5 * 10**8
    print("Edge Case 4 (10^9 scale): Passed")

    # Standard Examples
    assert minEatingSpeed([3, 6, 7, 11], 8) == 4
    print("Example 1 ([3, 6, 7, 11], h=8): Passed -> k=4")

    assert minEatingSpeed([30, 11, 23, 4, 20], 5) == 30
    print("Example 2 ([30, 11, 23, 4, 20], h=5): Passed -> k=30")

    assert minEatingSpeed([30, 11, 23, 4, 20], 6) == 23
    print("Example 3 ([30, 11, 23, 4, 20], h=6): Passed -> k=23")

    print("\nAll edge cases and examples passed cleanly.")