"""
怕吗?
"""
from typing import List


class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        """ 643.子数组最大平均数I """
        ## 想到滑窗
        if len(nums) == 1:
            return nums[0]
        curAvg = maxAvg = sum(nums[:k])
        for i in range(k, len(nums)):
            curAvg = curAvg - nums[i - k] + nums[i]
            maxAvg = max(maxAvg, curAvg)
        return maxAvg / k
    
    def lengthOfLIS(self, nums: List[int]) -> int:
        """ 300.最长递增子序列 """
        n = len(nums)
        dp = [1] * n        # dp[i] 以nums[i]为结尾的最长递增子序列长度
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)
    
    def findLengthOfLCIS(self, nums: List[int]) -> int:
        """ 674.最长连续递增序列 \n
            300.最长递增子序列 不要求连续, 本题要求连续 """
        n = len(nums)
        dp = [1] * n        # dp[i] 以nums[i]结尾的最长连续递增子序列的长度
        for i in range(1, n):
            if nums[i] > nums[i - 1]:
                dp[i] = dp[i - 1] + 1
        return max(dp)

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        """ 15.三数之和 """
        nums.sort()
        res = []
        for i in range(len(nums) - 2):
            if i > 0 and nums[i - 1] == nums[i]:
                continue
            l, r = i + 1, len(nums) - 1
            while l < r:
                _tmp = [nums[i], nums[l], nums[r]]
                _sum = sum(_tmp)
                if _sum == 0:
                    res.append(_tmp[:])
                    while l < r and nums[l + 1] == nums[l]:
                        l += 1
                    while l < r and nums[r - 1] == nums[r]:
                        r -= 1
                    l += 1
                    r -= 1
                elif _sum < 0:
                    l += 1
                else:
                    r -= 1
        return res
    
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        """ 16.最接近的三数之和 \n
            与 15.三数之和 基本相同 """
        nums.sort()
        res = sum(nums[:3])
        for i in range(len(nums) - 2):
            if i > 0 and nums[i - 1] == nums[i]:
                continue
            l, r = i + 1, len(nums) - 1
            while l < r:
                _sum = nums[i] + nums[l] + nums[r]
                res = _sum if abs(target - _sum) < abs(target - res) else res
                if _sum == target:
                    return res
                elif _sum < target:
                    l += 1
                else:
                    r -= 1
        return res
    
    def QuickSort(self, nums, low, high):
        """ 快排 """
        def partition(low, high):
            pivot = nums[low]
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
    
    def myPow(self, x: float, n: int) -> float:
        """ 50.Pow(x,n) """
        if n < 0:
            x = 1 / x
            n = -n
        res = 1
        while n:
            if n & 1:
                res *= x
            n >>= 1
            x *= x
        return res
    
    def findKthPositive(self, arr: List[int], k: int) -> int:
        """ 1539.第k个缺失的正整数 """
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
        """ 227.基本计算器II """
        s = s.replace(' ', '')

        sign = '+'
        num = 0
        stack = []
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
                else:
                    stack.append(int(stack.pop() / num))
                
                sign = s[i]
                num = 0
        return sum(stack)
    
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        """ 595.岛屿的最大面积 """
        m = len(grid)
        n = len(grid[0])
        visited = [[False] * n for _ in range(m)]
        maxArea = 0

        def dfs(x, y):
            visited[x][y] = True
            grid[x][y] = 0
            nonlocal count
            count += 1
            for _dr in [-1, 0], [1, 0], [0, -1], [0, 1]:
                nextX = x + _dr[0]
                nextY = y + _dr[1]
                if nextX < 0 or nextX >= m or nextY < 0 or nextY >= n:
                    continue
                if visited[nextX][nextY] == False and grid[nextX][nextY] == 1:
                    dfs(nextX, nextY)

        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1 and visited[i][j] == False:
                    count = 0
                    dfs(i, j)
                    maxArea = max(maxArea, count)
        return maxArea
    
    def numIslands(self, grid: List[List[str]]) -> int:
        """ 200.岛屿数量 """
        # res = 0
        # m, n = len(grid), len(grid[0])
        # vis = [[False] * n for _ in range(m)]

        # def dfs(x, y):
        #     vis[x][y] = True
        #     grid[x][y] == '0'
        #     for _dr in [-1, 0], [1, 0], [0, -1], [0, 1]:
        #         nextX = x + _dr[0]
        #         nextY = y + _dr[1]
        #         if nextX < 0 or nextX >= m or nextY < 0 or nextY >= n:
        #             continue
        #         if not vis[nextX][nextY] and grid[nextX][nextY] == '1':
        #             dfs(nextX, nextY)

        # for i in range(m):
        #     for j in range(n):
        #         if not vis[i][j] and grid[i][j] == '1':
        #             res += 1
        #             dfs(i, j)
        # return res

        ## bfs 还是推荐上面的
        m, n = len(grid), len(grid[0])
        vis = [[False] * n for _ in range(m)]
        res = 0

        def bfs(x, y):
            from collections import deque
            
            queue = deque([[x, y]])
            grid[x][y] = '0'
            vis[x][y] = True
            while queue:
                x, y = queue.popleft()
                for _dr in [-1, 0], [1, 0], [0, -1], [0, 1]:
                    nextX, nextY = x + _dr[0], y + _dr[1]
                    if nextX < 0 or nextX >= m or nextY < 0 or nextY >= n:
                        continue
                    if not vis[nextX][nextY] and grid[nextX][nextY] == '1':
                        queue.append([nextX, nextY])
                        grid[nextX][nextY] = '0'
                        vis[nextX][nextY] = True
        
        for i in range(m):
            for j in range(n):
                if not vis[i][j] and grid[i][j] == '1':
                    res += 1
                    bfs(i, j)
        return res
    
    def numEnclaves(self, grid: List[List[int]]) -> int:
        """ 1020.飞地的数量 """
        ## 题意不好理解, 直接看代码
        m, n = len(grid), len(grid[0])

        def dfs(x, y):
            grid[x][y] = 0
            for _dr in [-1, 0], [1, 0], [0, -1], [0, 1]:
                nextX = x + _dr[0]
                nextY = y + _dr[1]
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
        return sum([sum(row) for row in grid])

if __name__ == '__main__':
    sl = Solution()

    nums = [4,0,4,3,3]
    k = 5
    print(sl.findMaxAverage(nums, k))