from typing import List


class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        """
        200.岛屿数量
        2023.05.17 中等
        题解：dfs
        :param grid:
        :return:
        """
        def dfs(grid, x, y):
            grid[x][y] = 0
            nr = len(grid)
            nc = len(grid[0])
            for x, y in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                if 0 <= x < nr and 0 <= y < nc and grid[x][y] == "1":
                    dfs(grid, x, y)

        nr = len(grid)
        if nr == 0:
            return 0
        nc = len(grid[0])
        num_islands = 0
        for x in range(nr):
            for y in range(nc):
                if grid[x][y] == "1":
                    num_islands += 1
                    dfs(grid, x, y)
        return num_islands