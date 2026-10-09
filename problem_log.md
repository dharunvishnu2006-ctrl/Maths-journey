| Date | Problem | Pattern | Time | Hints | Key Insight |
|---|---|---|---|---|---|
| 2026-10-08 | Longest Consecutive Sequence (LC 128) | Hash Set / Array | 42 mins | Rung 0 | Only start sequence counting when `num - 1` is not in the set to guarantee O(n). |
| 2026-10-09 | Container With Most Water (LC 11) | Two Pointers (Opposite Ends) | 50 mins | Rung 0 | The shorter side is the bottleneck; keeping it while shrinking width can never increase area. |

| 2026-10-09 | LC 3: Longest Substring Without Repeating Characters | Sliding Window (Two Pointers + Hash Map) | 36 mins | Rung 0 | Verified edge cases first (`""`, `" "`). Beware indexing off-by-ones when fast-forwarding `left`: defaulting unseen keys in `.get()` requires `-1` rather than `0` to avoid incorrectly shifting the left boundary forward on the initial character. |