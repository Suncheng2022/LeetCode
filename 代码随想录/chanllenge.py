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

# res = numIslands_1()
# print(res)

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

# if __name__ == "__main__":
#     res = numIslands_1()
#     print(res)

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

# if __name__ == "__main__":
#     res = numIsland_3()
#     print(res)

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

# if __name__ == "__main__":
#     res = numIsland_3()
#     print(res)

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

# if __name__ == "__main__":
#     print(numIsland_4())

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

# if __name__ == "__main__":
#     print(numIsland_4())

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
    
    def func(s):
        """
        新东方一面
        LeetCode 227 基本计算器 II、772 等
        1.按符号分开, 数字入队列
        2.遍历符号, 按优先级计算
        """
        stack = []
        sign = '+'
        num = 0
        for i in range(len(s)):
            # 若是数字
            if s[i].isdigit():
                num = num * 10 + int(s[i])
            # 若是符号
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
                    stack.append(int(_top / num))   # 为什么取整?
            
                sign = s[i]
                num = 0

        return sum(stack)

    def calculate(self, s: str) -> int:
        """ 227.基本计算器II """
        s = s.replace(' ', '')
        stack = []
        sign = '+'
        num = 0
        for i in range(len(s)):
            if s[i].isdigit():
                num = num * 10 + int(s[i])
            
            if not s[i].isdigit() or i == len(s) - 1:
                if sign == '+':
                    stack.append(num)
                elif sign == '-':
                    stack.append(-num)
                elif sign == '*':
                    stack.append(stack.pop() * num)
                elif sign == '/':
                    stack.append(int(stack.pop() / num))
                sign = s[i]
                num = 0
        return sum(stack)

    def findKthPositive(self, arr: List[int], k: int) -> int:
        """ 1539.第k个缺失的正整数 \n
            字节视频搜索一面 """
        cur = 1         # 当前遍历的数字, 题目已说明从1开始
        miss_count = 0  # 缺失数字计数
        i = 0           # 当前遍历的下标
        while True:
            if i < len(arr) and arr[i] == cur:
                i += 1
            else:
                miss_count += 1
                if miss_count == k:
                    return cur
            cur += 1
    
    """
    字节视频搜索一面 \n
    问题:20个苹果，分给5个人，每人至少一个苹果，多少种方法？ \n
    每人先分1个, 用掉5个.
    剩下15个, 怎么分, 相当于将15个苹果分为5份, 有的份可以为0 --> 往15个苹果中间插4个板子, 从 15 + 4 = 19 个位置里，挑 4 个位置放板子, 即C_19^4
    """