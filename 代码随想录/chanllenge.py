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
    """ 1020.飞地的数量 \n
        岛屿问题(四):孤岛的总面积 \
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
        """ 659.岛屿的最大面积  \n
            岛屿问题（三）：最大面积 """
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
        """ 同 659.岛屿的最大面积\n
            岛屿问题（四）：孤岛的总面积 """
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
        """ 这个题不太好理解, skip it. \n
            岛屿问题（五）：沉没孤岛 \n
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

    def myPow(self, x: float, n: int) -> float:
        """
        50.Pow(x, n) \n
        Shopee图搜一面
        """
        ## 优化. 将幂转为二进制
        if n == 0:
            return 1
        if n < 0:
            x, n = 1 / x, -n
        res = 1
        while n:
            if n & 1:
                res *= x
            n >>= 1
            x *= x      # 如果写成 x = x ** 2, 会不会更明了一点
        return res

        ## 我的实现--超时
        # if n == 0:
        #     return 1
        # flag = 1 if n > 0 else -1
        # n = abs(n)
        # res = 1
        # for _ in range(n):
        #     res *= x
        # return res if flag == 1 else 1 / res
    
    """
    新东方技术三面:
        有12颗外观相同的小球, 其中1颗重量不同, 给你一个天平, 没有砝码, 最多称3次找出那颗小球, 并判断出它重还是轻?
    信息论视角: 每次天平称重有3种结果--1.左边重 2.右边重 3.平衡. 题目要求的3次称重可以覆盖3**3=27种情况.
            一共12颗小球, 每个小球有 重 或 轻 2种情况, 一共有12*2=24种情况, 理论上题目要求可以达到.
    关键：每次称重要尽可能均分可能性，获取最大信息量。
    eg: 
        第一次称重, 左:1 2 3 4 右:5 6 7 8 若平衡, 则异常球在余下的球中. 第二次称重, 左: 1~8任选3颗 右:9 10 11, 不平衡--则知异常球轻还是重 平衡--则12是异常球 第三次, 左: 9 右:10. 解了
    """

    def QuickSort(self, nums, low, high):
        """ 快速排序 \n
            虾皮一面 寄, 没写出来. 刷题还是少呀 """
        # def partition(low, high):
        #     ''' 基准划分 填坑法 \n
        #         要点: 
        #             1.pivot = nums[low]说明nums[low]已经是坑了, 所以先从右往左找 先填上nums[low]
        #             2.最后pivot放在low或high都对, 因为while退出条件 '''
        #     pivot = nums[low]       # 在一次划分期间固定基准
        #     while low < high:
        #         while low < high and nums[high] >= pivot:   # 先从右往左找
        #             high -= 1
        #         nums[low] = nums[high]
        #         while low < high and nums[low] <= pivot:
        #             low += 1
        #         nums[high] = nums[low]
        #     nums[low] = pivot       # 基准归位
        #     return low
        
        # if low < high:
        #     k = partition(low, high)
        #     self.QuickSort(nums, low, k - 1)
        #     self.QuickSort(nums, k + 1, high)

        ## Again
        def partition(low, high):
            pivot = nums[low]       # 一次划分前, 先确定基准
            while low < high:
                while low < high and nums[high] >= pivot:
                    high -= 1
                nums[low] = nums[high]
                while low < high and nums[low] <= pivot:
                    low += 1
                nums[high] = nums[low]
            nums[low] = pivot
            return low
        
        if low < high:
            k = partition(low, high)
            self.QuickSort(nums, low, k - 1)
            self.QuickSort(nums, k + 1, high)


    def threeSumClosest(self, nums: List[int], target: int) -> int:
        """ 16.最接近的三数之和 \n
            虾皮一面 寄, 没做过 """
        nums.sort()
        res = sum(nums[:3])     # 要给一个真实的基准
        for i in range(len(nums) - 2):
            start, end = i + 1, len(nums) - 1
            while start < end:
                sum = nums[i] + nums[start] + nums[end]
                res = sum if abs(target - sum) < abs(target - res) else res
                if sum < target:
                    start += 1
                elif sum == target:
                    return sum
                else:
                    end -= 1
        return res

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        """ 15.三数之和 """
        nums.sort()
        res = []
        for i in range(len(nums) - 2):
            start, end = i + 1, len(nums) - 1
            if i > 0 and nums[i - 1] == nums[i]:        # 去重
                continue
            while start < end:
                _tmp = [nums[i], nums[start], nums[end]]
                _sum = sum(_tmp)
                if _sum == 0:
                    res.append(_tmp)
                    while start < end and nums[start] == nums[start + 1]:   # 跳出情况一: 已经用start < end限制了, 跳出时即便start == end, 但最外层的for会兜住
                        start += 1
                    while start < end and nums[end - 1] == nums[end]:
                        end -= 1
                    start += 1                                              # 跳出情况二: 跳出时nums[start] != nums[start + 1], 但此时nums[start]依然是上一个重复区间, 所以要右移彻底跳出去
                    end -= 1
                elif _sum < 0:
                    start += 1
                else:
                    end -= 1
        return res
        
