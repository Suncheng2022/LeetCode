from typing import List


class Solution:
    def climbStairs(self, n: int) -> int:
        """
        70.爬楼梯
        2023.07.19 简单
        题解：斐波那契数列
        """
        dp = [0 for _ in range(n + 1)]
        dp[0] = 1
        dp[1] = 1
        for i in range(2, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[-1]

    def fib(self, n: int) -> int:
        """
        509.斐波那契数
        2023.07.19 简单
        """
        if n == 0:
            return 0
        dp = [0 for _ in range(n + 1)]
        dp[0] = 0
        dp[1] = 1
        for i in range(2, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[n]

    def tribonacci(self, n: int) -> int:
        """
        1137.第N个泰波那契数
        2023.07.19 简单
        """
        if n == 0:
            return 0
        elif n <= 2:
            return 1
        dp = [0 for _ in range(n + 1)]
        dp[0] = 0
        dp[1] = 1
        dp[2] = 1
        for i in range(3, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2] + dp[i - 3]
        return dp[-1]

    def minCostClimbingStairs(self, cost: List[int]) -> int:
        """
        746.使用最小花费爬楼梯
        2023.07.19 简单
        题解：
        """
        n = len(cost)
        dp = [0 for _ in range(n + 1)]      # 到达i时的最小花费(只计算到达i所经过的元素，不包括i)
        dp[0] = dp[1] = 0       # 只计算到达i所经过的元素，不包括i
        for i in range(2, n + 1):
            dp[i] = min(dp[i - 1] + cost[i - 1],    # 从i-1到达i，就要计算经过i-1的费用
                        dp[i - 2] + cost[i - 2])    # 从i-2到达i，就要计算经过i-2的费用
        return dp[n]

    def rob(self, nums: List[int]) -> int:
        """
        198.打家劫舍
        2023.07.19 中等
        """
        # 重写一遍
        n = len(nums)
        dp = [0 for _ in range(n + 1)]      # 前i间房屋能偷到的最大金额
        dp[1] = nums[0]
        for i in range(2, n + 1):       # 计算前i间房屋能偷到多少，注意与nums[i]的i区分
            dp[i] = max(nums[i - 1] + dp[i - 2], dp[i - 1])
        return dp[n]

        # n = len(nums)
        # dp = [0 for _ in range(n + 1)]
        # dp[1] = nums[0]
        # for i in range(2, n + 1):
        #     dp[i] = max(dp[i - 1], dp[i - 2] + nums[i - 1])
        # return dp[n]

    def deleteAndEarn(self, nums: List[int]) -> int:
        """
        740.删除并获得点数
        2023.07.21 中等
        题解：类似 198.打家劫舍
        """
        from collections import Counter

        num2counts_dict = Counter(nums)
        total = [0] * (max(nums) + 1)
        for num, counts in num2counts_dict.items():
            total[num] = num * counts
        # 后面和打家劫舍一样了
        dp = [0 for _ in range(len(total) + 1)]
        dp[1] = total[0]
        for i in range(2, len(total) + 1):
            dp[i] = max(total[i - 1] + dp[i - 2], dp[i - 1])
        return dp[-1]

    def uniquePaths(self, m: int, n: int) -> int:
        """
        62.不同路径
        2023.07.21 中等
        """
        dp = [[1] * n] + [[1] + [0] * (n - 1) for _ in range(m - 1)]
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[-1][-1]

    def minPathSum(self, grid: List[List[int]]) -> int:
        """
        64.最小路径和
        2023.07.21 中等
        """
        m = len(grid)
        n = len(grid[0])
        for i in range(m):
            for j in range(n):
                if i == 0 and j - 1 >= 0:
                    grid[i][j] += grid[i][j - 1]
                elif j == 0 and i - 1 >= 0:
                    grid[i][j] += grid[i - 1][j]
                elif i - 1 >= 0 and j - 1 >= 0:       # 这个条件注意，[0][0]就能走进来
                    grid[i][j] += min(grid[i][j - 1], grid[i - 1][j])
        return grid[-1][-1]

    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        """
        63.不同路径II
        2023.07.21 中等
        题解：看下之前记录的代码，思路没那么复杂哦
        """
        m = len(obstacleGrid)
        n = len(obstacleGrid[-1])
        dp = [[0] * n for _ in range(m)]
        dp[0][0] = 1
        for i in range(m):
            for j in range(n):
                if obstacleGrid[i][j]:      # 遍历到障碍物
                    dp[i][j] = 0            # 到障碍物网格路径数量为0
                    continue                # 这也不能忘，遇到障碍物也是处理了此处的dp
                if i == 0 and j - 1 >= 0:
                    dp[i][j] = dp[i][j - 1]
                elif i - 1 >= 0 and j == 0:
                    dp[i][j] = dp[i - 1][j]
                elif i - 1 >= 0 and j - 1 >= 0:
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[-1][-1]

    def minimumTotal(self, triangle: List[List[int]]) -> int:
        """
        120.三角形最小路径和
        2023.07.21 中等
        """
        # 自己改为了本地修改！牛逼啦！
        for i in range(1, len(triangle)):       # 三角形 第一行只有1个元素
            for j in range(len(triangle[i])):
                if j == 0:
                    triangle[i][j] += triangle[i - 1][j]
                elif j == len(triangle[i]) - 1:
                    triangle[i][j] += triangle[i - 1][-1]   # 此处两个j处都为-1
                else:
                    triangle[i][j] += min(triangle[i - 1][j - 1], triangle[i - 1][j])
        return min(triangle[-1])

        # dp = [[0] * len(ls) for ls in triangle]
        # dp[0][0] = triangle[0][0]
        # for i in range(1, len(triangle)):       # 猜测 三角形第一行只1个元素
        #     for j in range(len(triangle[i])):
        #         # 特殊处理当前处理行的 第一个 和 最后一个
        #         if j == len(triangle[i]) - 1:
        #             dp[i][j] = dp[i - 1][j - 1] + triangle[i][j]
        #         elif j - 1 >= 0:
        #             dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j]) + triangle[i][j]
        #         elif j - 1 < 0:     # j < 1
        #             dp[i][j] = dp[i - 1][0] + triangle[i][j]
        # return min(dp[-1])


if __name__ == "__main__":
    sl = Solution()

    triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
    print(sl.minimumTotal(triangle))
