"""
04.04   面试题目
"""
def numIslands_1():
    """ 岛屿问题（一）：岛屿数量 深度搜索 """
    m, n = map(int, input().split())
    grid = [list(map(int, input().split())) for _ in range(m)]
    res = 0

    def dfs(x, y):
        grid[x][y] = 0
        visited[x][y] = True
        for dir in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
            nextX, nextY = x + dir[0], y + dir[1]
            if nextX < 0 or nextX >= m or nextY < 0 or nextY >= n: 
                continue
            if visited[nextX][nextY] == False and grid[nextX][nextY] == 1:
                dfs(nextX, nextY)

    visited = [[False] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            if visited[i][j] == False and grid[i][j] == 1:
                res += 1
                dfs(i, j)       # 标记邻接区域
    return res

res = numIslands_1()
print(res)

def numIslands_1():
    """ 岛屿问题（一）：岛屿数量 广度搜索 """
    m, n = map(int, input().split())
    grid = [list(map(int, input().split())) for _ in range(m)]

    visited = [[False] * n for _ in range(m)]
    res = 0

    from collections import deque
    def bfs(x, y):
        visited[x][y] = True
        que = deque()
        que.append([x, y])
        while que:
            x, y = que.popleft()
            for dir in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
                nextX, nextY = x + dir[0], y + dir[1]
                if nextX < 0 or nextX >= m or nextY < 0 or nextY >= n:
                    continue
                if not visited[nextX][nextY] and grid[nextX][nextY] == 1:
                    visited[nextX][nextY] = True
                    que.append([nextX, nextY])
        
    for i in range(m):
        for j in range(n):
            if not visited[i][j] and grid[i][j] == 1:
                res += 1
                bfs(i, j)
    return res

if __name__ == "__main__":
    res = numIslands_1()
    print(res)

def numIsland_3():
    """ 岛屿问题(三): 岛屿的最大面积\n
        dfs解法 """
    m, n = map(int, input().split())
    grid = [list(map(int, input().split())) for _ in range(m)]

    visited = [[False] * n for _ in range(m)]
    max_res = 0

    def dfs(x, y):
        visited[x][y] = True
        for dir in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
            nextX, nextY = x + dir[0], y + dir[1]
            if nextX < 0 or nextX >= m or nextY < 0 or nextY >= n:
                continue
            if not visited[nextX][nextY] and grid[nextX][nextY] == 1:
                nonlocal cur_count
                cur_count += 1
                dfs(nextX, nextY)

    for i in range(m):
        for j in range(n):
            if not visited[i][j] and grid[i][j] == 1:
                cur_count = 1
                dfs(i, j)
                max_res = max(max_res, cur_count)
    return max_res

if __name__ == "__main__":
    res = numIsland_3()
    print(res)

def numIsland_3():
    """ 岛屿问题(三): 岛屿的最大面积\n
        bfs解法 """
    m, n = map(int, input().split())
    grid = [list(map(int, input().split())) for _ in range(m)]

    visited = [[False] * n for _ in range(m)]
    max_res = 0

    from collections import deque
    def bfs(x, y):
        visited[x][y] = True
        que = deque()
        que.append([x, y])
        while que:
            x, y = que.popleft()
            for dir in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
                nextX, nextY = x + dir[0], y + dir[1]
                if nextX < 0 or nextX >= m or nextY < 0 or nextY >= n:
                    continue
                if not visited[nextX][nextY] and grid[nextX][nextY] == 1:
                    visited[nextX][nextY] = True
                    nonlocal cur_count
                    cur_count += 1
                    que.append([nextX, nextY])

    for i in range(m):
        for j in range(n):
            if not visited[i][j] and grid[i][j] == 1:
                cur_count = 1
                bfs(i, j)
                max_res = max(max_res, cur_count)
    return max_res

if __name__ == "__main__":
    res = numIsland_3()
    print(res)

def numIsland_4():
    """ 岛屿问题(四):孤岛的总面积 \
        dfs """
    m, n = map(int, input().split())
    grid = [list(map(int, input().split())) for _ in range(m)]

    def dfs(x, y):
        grid[x][y] = 0
        for dir in [[-1, 0], [1, 0], [0, 1], [0, -1]]:
            nextX, nextY = x + dir[0], y + dir[1]
            if nextX < 0 or nextX >= m or nextY < 0 or nextY >= n:
                continue
            if grid[nextX][nextY] == 0:
                continue
            dfs(nextX, nextY)
    
    # 处理左右边界
    for i in range(m):
        if grid[i][0] == 1:
            dfs(i, 0)
        if grid[i][n - 1] == 1:
            dfs(i, n - 1)
    # 处理上下边界
    for j in range(n):
        if grid[0][j] == 1:
            dfs(0, j)
        if grid[m - 1][j] == 1:
            dfs(m - 1, j)
    # 计算孤岛总面积
    res = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                res += 1
    return res

if __name__ == "__main__":
    print(numIsland_4())

def numIsland_4():
    """ 岛屿问题(四):孤岛的总面积 \
        bfs """
    m, n = map(int, input().split())
    grid = [list(map(int, input().split())) for _ in range(m)]

    from collections import deque
    def bfs(x, y):
        que = deque([[x, y]])
        grid[x][y] = 0
        while que:
            x, y = que.popleft()
            for dir in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
                nextX, nextY = x + dir[0], y + dir[1]
                if nextX < 0 or nextX >= m or nextY < 0 or nextY >= n:
                    continue
                if grid[nextX][nextY] == 1:
                    que.append([nextX, nextY])
                    grid[nextX][nextY] = 0

    # 处理左右边界
    for i in range(m):
        if grid[i][0] == 1:
            bfs(i, 0)
        if grid[i][n - 1] == 1:
            bfs(i, n - 1)
    # 处理上下边界
    for j in range(n):
        if grid[0][j] == 1:
            bfs(0, j)
        if grid[m - 1][j] == 1:
            bfs(m - 1, j)
    
    res = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                res += 1
    return res

if __name__ == "__main__":
    print(numIsland_4())

# ========================= 第一次实现 ==============================
from typing import List

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        """ 200.岛屿数量 \n
            dfs """
        m, n = len(grid), len(grid[0])
        visited = [[False] * n for _ in range(m)]
        res = 0

        def dfs(i, j):
            """ 将与[i, j]相关联的陆地都标记 """
            for dir in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
                nextX, nextY = i + dir[0], j + dir[1]
                if nextX < 0 or nextX >= m or nextY < 0 or nextY >= n: 
                    continue
                if visited[nextX][nextY] == False and grid[nextX][nextY] == '1':
                    visited[nextX][nextY] = True
                    dfs(nextX, nextY)

        for i in range(m):
            for j in range(n):
                if visited[i][j] == False and grid[i][j] == '1':
                    res += 1
                    dfs(i, j)
        return res
    
    def numIslands_bfs(self, grid: List[List[str]]) -> int:
        """ 岛屿数量 bfs """
        m, n = len(grid), len(grid[0])
        visited = [[False] * n for _ in range(m)]
        res = 0

        from collections import deque
        def bfs(x, y):
            visited[x][y] = True        # 广搜 在里面置true
            que = deque()
            que.append((x, y))
            while que:
                x, y = que.popleft()
                for dir in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
                    nextX, nextY = x + dir[0], y + dir[1]
                    if nextX < 0 or nextX >= m or nextY < 0 or nextY >= n: 
                        continue
                    if visited[nextX][nextY] == False and grid[nextX][nextY] == '1':
                        visited[nextX][nextY] = True
                        que.append([nextX, nextY])

        for i in range(m):
            for j in range(n):
                if visited[i][j] == False and grid[i][j] == '1':
                    res += 1
                    bfs(i, j)
        return res
    
    def numIsland_area(self, grid: List[List[int]]) -> int:
        """ 岛屿问题（三）：最大面积 """
        m, n = len(grid), len(grid[0])
        visited = [[False] * n for _ in range(m)]
        max_area = 0
        count = 0

        def dfs(x, y):
            for dir in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
                nextX, nextY = x + dir[0], y + dir[1]
                if nextX < 0 or nextX >= m or nextY < 0 or nextY >= n:
                    continue
                if visited[nextX][nextY] == False and grid[nextX][nextY] == '1':
                    visited[nextX][nextY] = True
                    count += 1
                    dfs(nextX, nextY)

        for i in range(m):
            for j in range(n):
                if visited[i][j] == False and grid[i][j] == '1':
                    visited[i][j] = True
                    count = 1
                    dfs(i, j)
                    max_area = max(max_area, count)
        return max_area
    
    def numIsland_4(self, grid: List[List[int]]) -> int:
        """ 岛屿问题（四）：孤岛的总面积 """
        # 按ACM格式
        res = 0
        m, n = map(int, input().split())
        grid = []
        for _ in range(m):
            grid.append(list(map(int, input().split())))
        
        def dfs(x, y):
            nonlocal res
            res += 1
            for dir in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
                nextX, nextY = x + dir[0], y + dir[1]
                if nextX < 0 or nextX >= m or nextY < 0 or nextY >= n:
                    continue
                if grid[nextX][nextY] == 1:
                    grid[nextX][nextY] = 0
                    dfs(nextX, nextY)

        # 清除边界
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
        
        res = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    dfs(i, j)
        return res
    
    from typing import List
    def numIsland_5() -> List[List[int]]:
        """ 岛屿问题（五）：沉没孤岛 \n
            按照ACM格式输入 """
        m, n = map(int, input().split())
        grid = [list(map(int, input().split())) for _ in range(m)]

        def dfs(x, y):
            grid[x][y] = 2
            for dir in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
                nextX, nextY = x + dir[0], y + dir[1]
                if nextX < 0 or nextX >= m or nextY < 0 or nextY >= n:
                    continue
                if grid[nextX][nextY] in [0, 2]:
                    continue
                # elif grid[nextX][nextY] == 1:
                dfs(nextX, nextY)
        
        # 处理边界
        for i in range(m):
            if grid[i][0] == 1:
                dfs(i, 0)
            elif grid[i][n - 1] == 1:
                dfs(i, n - 1)
        for j in range(n):
            if grid[0][j] == 1:
                dfs(0, j)
            elif grid[m - 1][j] == 1:
                dfs(m - 1, j)
        # 沉没
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    grid[i][j] = 0
                elif grid[i][j] == 2:
                    grid[i][j] = 1
        return grid