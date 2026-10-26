| Date | Problem | Pattern | Time | Hints | Key Insight |
|---|---|---|---|---|---|
| 2026-10-08 | Longest Consecutive Sequence (LC 128) | Hash Set / Array | 42 mins | Rung 0 | Only start sequence counting when `num - 1` is not in the set to guarantee O(n). |
| 2026-10-09 | Container With Most Water (LC 11) | Two Pointers (Opposite Ends) | 50 mins | Rung 0 | The shorter side is the bottleneck; keeping it while shrinking width can never increase area. |

| 2026-10-09 | LC 3: Longest Substring Without Repeating Characters | Sliding Window (Two Pointers + Hash Map) | 36 mins | Rung 0 | Verified edge cases first (`""`, `" "`). Beware indexing off-by-ones when fast-forwarding `left`: defaulting unseen keys in `.get()` requires `-1` rather than `0` to avoid incorrectly shifting the left boundary forward on the initial character. |

| 2026-10-12 | Daily Temperatures (LC 739) | Monotonic Stack | 38 mins | Rung 0 | Solved with monotonic decreasing index stack; skipped standalone prediction step before running |

| Date | Problem | Pattern | Time | Rung | What I learned |
| --- | --- | --- | --- | --- | --- |
| 2026-10-13 | LC 875 Koko Eating Bananas | Binary Search on Answer | 48 min | 0 | When total hours decrease monotonically as speed increases, search the bounded answer space $[1, \max(\text{piles})]$. Use $(p + k - 1) // k$ to compute ceiling strictly with integer arithmetic to avoid IEEE 754 float precision drift on large values. Write plan comments *before* writing code so they guide the build rather than document it after. |

| Date | Problem | Pattern | Time | Rung | What I learned |
| --- | --- | --- | --- | --- | --- |
| 2026-10-14 | Remove Nth Node From End of List (LeetCode 19) | Two Pointers (Fixed-Gap / Fast & Slow) + Dummy Sentinel | 34m | 0 | Using a dummy head unified the edge cases when deleting the true head node, and spacing fast $(n+1)$ ahead positions slow exactly on the predecessor node for an $O(1)$-space, single-pass deletion. |

| Date | Problem | Pattern | Time | Rung | What I learned |
| --- | --- | --- | --- | --- | --- |
| 2026-10-16 | Binary Tree Level Order Traversal (LeetCode 102) | BFS / Level-Order Traversal with Queue (`collections.deque`) | 49m | Rung 2 | Freezing `level_size = len(queue)` acts as a bookmark to process one full tier before child pushes bleed into it, while `deque.popleft()` ensures an $O(N)$ runtime instead of the $O(N^2)$ shifting penalty of `list.pop(0)`. |

| Date | Problem | Pattern | Time | Rung | What I learned |
| --- | --- | --- | --- | --- | --- |
| 2026-10-26 | LeetCode 200: Number of Islands | Graph / DFS Flood Fill | 15 mins | Rung 1 | When you find land ('1'), sink the entire connected island to '0' with DFS across the 4 orthogonal directions so it is never double-counted. Diagonals do not connect. |