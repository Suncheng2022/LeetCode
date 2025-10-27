""" 10.25 随便写写 """
from typing import List

class Solution:
    def findKthPositive(self, arr: List[int], k: int) -> int:
        cur = 1
        miss_count = 0
        i = 0
        while True:
            if i < len(arr) and arr[i] == cur:
                i += 1
            else:
                miss_count += 1
                if miss_count == k:
                    return cur
            cur += 1

    def calculate(self, s: str) -> int:
        s = s.replace(' ', '')

        stack = []
        num = 0
        sign = '+'
        for i in range(len(s)):
            if s[i].isdigit():
                num = num * 10 + int(s[i])
            if not s[i].isdigit() or i == len(s) - 1:
                if sign == '+':
                    stack.append(num)
                elif sign == '-':
                    stack.append(-num)
                elif sign == '*':
                    _top = stack.pop()
                    stack.append(_top * num)
                elif sign == '/':
                    _top = stack.pop()
                    stack.append(int(_top / num))
                sign = s[i]
                num = 0
        return sum(stack)
            
    def numIslands(self, grid: List[List[str]]) -> int:
        """ 200.岛屿数量 """
        m, n = len(grid), len(grid[0])
        visited = [[False] * n for _ in range(m)]
        res = 0

        def dfs(i, j):
            """ 标记所有与[i,j]相连的陆地 """
            visited[i][j] = True
            for dr in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
                nextX, nextY = i + dr[0], j + dr[1]
                if nextX < 0 or nextX >= m or nextY < 0 or nextY >= n:
                    continue
                if visited[nextX][nextY] == False and grid[nextX][nextY] == '1':
                    dfs(nextX, nextY)

        for i in range(m):
            for j in range(n):
                if visited[i][j] == False and grid[i][j] == '1':
                    res += 1
                    dfs(i, j)
        return res
        
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        """ 695.岛屿的最大面积 """
        m, n = len(grid), len(grid[0])
        visited = [[False] * n for _ in range(m)]
        max_count = 0
        
        def dfs(i, j):
            """ 标记所有与[i,j]相连的岛屿单元格 """
            nonlocal count
            visited[i][j] = True
            count += 1
            for dr in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
                nextX, nextY = i + dr[0], j + dr[1]
                if nextX < 0 or nextX >= m or nextY < 0 or nextY >= n:
                    continue
                if visited[nextX][nextY] == False and grid[nextX][nextY] == 1:
                    dfs(nextX, nextY)

        for i in range(m):
            for j in range(n):
                if visited[i][j] == False and grid[i][j] == 1:
                    count = 0
                    dfs(i, j)
                    max_count = max(max_count, count)
        return max_count
    
    def numEnclaves(self, grid: List[List[int]]) -> int:
        """ 1020.飞地的数量 """
        m, n = len(grid), len(grid[0])

        def dfs(x, y):
            grid[x][y] = 0
            for dr in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
                nextX, nextY = x + dr[0], y + dr[1]
                if nextX < 0 or nextX >= m or nextY < 0 or nextY >= n:
                    continue
                if grid[nextX][nextY] == 1:
                    dfs(nextX, nextY)

        for i in range(m):
            if grid[i][0] == 1:
                dfs(i, 0)
            if grid[i][n - 1] == 1:
                dfs(i, n - 1)
        for j in range(n):
            if grid[0][j] == 1:
                dfs(0, j)
            if grid[m - 1][j] == 1:
                dfs(m - 1, j)
        return sum([sum(l) for l in grid])


        ## 超时
        # m, n = len(grid), len(grid[0])

        # from collections import deque
        # def bfs(x, y):
        #     queue = deque([[x, y]])
        #     while queue:
        #         x, y = queue.popleft()
        #         grid[x][y] = 0
        #         for dr in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
        #             nextX, nextY = x + dr[0], y + dr[1]
        #             if nextX < 0 or nextX >= m or nextY < 0 or nextY >= n:
        #                 continue
        #             if grid[nextX][nextY] == 1:
        #                 queue.append([nextX, nextY])

        # for i in range(m):
        #     if grid[i][0] == 1:
        #         bfs(i, 0)
        #     if grid[i][n - 1] == 1:
        #         bfs(i, n - 1)
        
        # for j in range(n):
        #     if grid[0][j] == 1:
        #         bfs(0, j)
        #     if grid[m - 1][j] == 1:
        #         bfs(m - 1, j)
        
        # return sum([sum(row) for row in grid])
            
if __name__ == '__main__':
    sl = Solution()
    arr = [2,3,4,7,11]
    k = 5
    print(sl.findKthPositive(arr, k))