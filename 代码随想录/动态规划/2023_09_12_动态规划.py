from typing import List


class Solution:
    def fib(self, n: int) -> int:
        # dp = [0] * (n + 1)     # dp[i] 索引第i个斐波那契数是dp[i]
        # if n == 0:      # 根据题意，记得特殊处理n=0
        #     return dp[-1]
        # dp[0] = 0
        # dp[1] = 1
        # for i in range(2, n + 1):
        #     dp[i] = dp[i - 1] + dp[i - 2]
        # return dp[-1]

        # 优化下空间复杂度
        if n == 0:
            return 0
        dp = [0, 1]     # 就用dp长度为2
        for i in range(2, n + 1):
            f = sum(dp)
            dp[0] = dp[1]
            dp[1] = f
        return dp[-1]

    def climbStairs(self, n: int) -> int:
        """ 70.爬楼梯 """
        if n == 1:      # 特殊处理n=1的情况
            return 1
        dp = [0] * (n + 1)      # dp[i] 表示爬上第i阶楼梯有dp[i]种方法；《代码随想录》不考虑dp[0] 因为没意义，题目已说明n>=1为正整数
        dp[1] = 1
        dp[2] = 2
        for i in range(3, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[-1]

    def minCostClimbingStairs(self, cost: List[int]) -> int:
        """ 746.使用最小花费爬楼梯
            注意题目描述，从索引0或1起跳，到达索引0或1不花钱，从0或1跳就要花钱了 """
        n = len(cost)
        dp = [0] * (n + 1)      # dp[i] 到达索引0台阶所需最小花费; dp[0] dp[1]均初始化为0
        for i in range(2, n + 1):
            dp[i] = min(dp[i - 1] + cost[i - 1],    # 从索引i-1台阶跳到索引i台阶——到达索引i-1台阶最小花费dp[i-1]，起跳花费cost[i-1]
                        dp[i - 2] + cost[i - 2])    # 同上
        return dp[-1]

    def uniquePaths(self, m: int, n: int) -> int:
        """ 62.不同路径 """
        dp = [[1] * n] + [[1] + [0] * (n - 1) for _ in range(m - 1)]    # dp[i][j] 从坐标[0,0]走到[i,j]的路径数
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[-1][-1]

    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        """ 63.不同路径II """
        pass


if __name__ == '__main__':
    sl = Solution()

    m = 1
    n = 1
    print(sl.uniquePaths(m, n))
