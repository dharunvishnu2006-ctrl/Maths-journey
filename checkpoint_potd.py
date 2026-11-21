# UNDERSTAND:
# - Input: 2D grid of strings ("1" = land, "0" = water) of size R x C.
# - Output: Integer count of distinct islands.
# - Rules: Land connects horizontally and vertically (4 directions), NOT diagonally.
# - Edge cases: Empty grid, all water, all land, 1x1 grids, diagonal-only grids.

# PLAN:
# 1. Guard against an empty grid (return 0).
# 2. Loop through every cell (r, c) in the grid.
# 3. When a "1" is found, increment the island counter and trigger DFS.
# 4. In DFS, stop if out of bounds or on "0"; otherwise flip "1" -> "0" (sink it)
#    and recursively sink all 4 adjacent neighbors.
# 5. Return the total island count.

from typing import List


class Solution:

  def numIslands(self, grid: List[List[str]]) -> int:
    if not grid or not grid[0]:
      return 0

    rows, cols = len(grid), len(grid[0])
    islands = 0

    def dfs(r: int, c: int) -> None:
      if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == "0":
        return

      grid[r][c] = "0"

      dfs(r + 1, c)
      dfs(r - 1, c)
      dfs(r, c + 1)
      dfs(r, c - 1)

    for r in range(rows):
      for c in range(cols):
        if grid[r][c] == "1":
          islands += 1
          dfs(r, c)

    return islands


if __name__ == "__main__":
  sol = Solution()

  # Edge cases first
  assert sol.numIslands([]) == 0
  assert sol.numIslands([["0", "0"], ["0", "0"]]) == 0
  assert sol.numIslands([["1", "1"], ["1", "1"]]) == 1
  assert sol.numIslands([["1"]]) == 1
  assert sol.numIslands([["0"]]) == 0
  assert sol.numIslands([["1", "0", "1"], ["0", "1", "0"], ["1", "0", "1"]]) == 5

  # Standard case
  standard_grid = [
      ["1", "1", "0", "0", "0"],
      ["1", "1", "0", "0", "0"],
      ["0", "0", "1", "0", "0"],
      ["0", "0", "0", "1", "1"],
  ]
  assert sol.numIslands(standard_grid) == 3

  print("All test cases passed!")