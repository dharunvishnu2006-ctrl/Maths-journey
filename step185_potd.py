def numIslands(grid):
    # 1. If grid is empty or has no columns, return 0
    if not grid or not grid[0]:
        return 0

    # 2. Get rows (R) and cols (C), initialize island_count = 0
    R, C = len(grid), len(grid[0])
    island_count = 0

    # 3. Define a helper function (DFS) to sink connected land:
    def dfs(r, c):
        # - If out of bounds or current cell is '0', stop (return)
        if r < 0 or r >= R or c < 0 or c >= C or grid[r][c] == "0":
            return

        # - SINK THE LAND: change grid[r][c] from '1' to '0' so it is never counted again
        grid[r][c] = "0"

        # - Recursively visit 4 orthogonal neighbors: (up, down, left, right)
        dfs(r - 1, c)  # Up
        dfs(r + 1, c)  # Down
        dfs(r, c - 1)  # Left
        dfs(r, c + 1)  # Right

    # 4. Iterate through every cell (r, c) in the grid:
    for r in range(R):
        for c in range(C):
            # - If grid[r][c] == '1':
            if grid[r][c] == "1":
                # - Increment island_count by 1
                island_count += 1
                # - Call the helper to sink all connected '1's in this entire island
                dfs(r, c)

    # 5. Return island_count
    return island_count


# --- Test Cases ---

# Test 1: Example grid (Top-left block of 4, isolated center, bottom-right pair)
grid1 = [
    ["1", "1", "0", "0", "0"],
    ["1", "1", "0", "0", "0"],
    ["0", "0", "1", "0", "0"],
    ["0", "0", "0", "1", "1"],
]

# Test 2: Empty grid
grid2 = []

# Test 3: All-water grid
grid3 = [
    ["0", "0"],
    ["0", "0"],
]

print("Test 1 (Example):   ", numIslands(grid1))
print("Test 2 (Empty):     ", numIslands(grid2))
print("Test 3 (All water): ", numIslands(grid3))