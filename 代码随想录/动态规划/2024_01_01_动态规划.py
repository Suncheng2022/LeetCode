class Solution:
    def fib(self, n: int) -> int:
        """ 509.斐波那契数 """
        # dp = [0] * (n + 1)
        # if n < 1:                  # 处理边界
        #     return dp[n]
        # dp[0] = 0
        # dp[1] = 1
        # for i in range(2, n + 1):
        #     dp[i] = dp[i - 1] + dp[i - 2]
        # return dp[-1]

        # 优化一下空间复杂度
        if n <= 1:                  # 处理边界
            return n
        dp = [0] * 2                # dp初始化长度为2
        dp[0] = 0
        dp[1] = 1
        for i in range(2, n + 1):
            tmp = sum(dp)
            dp[0] = dp[1]
            dp[1] = tmp
        return dp[-1]

    def climbStairs(self, n: int) -> int:
        """ 70.爬楼梯
            递推公式 dp[i] = dp[i - 1] + dp[i - 2] """
        # if n == 1:
        #     return 1
        # dp = [0] * (n + 1)          # dp[i] 爬到第i阶楼梯有dp[i]种方法
        # dp[1] = 1                   # 初始化，不考虑dp[0]了，没实际意义，题目也已说明n>=1
        # dp[2] = 2
        # for i in range(3, n + 1):
        #     dp[i] = dp[i - 1] + dp[i - 2]
        # return dp[-1]

        # 优化空间复杂度
        # if n == 1:
        #     return 1
        # dp = [0] * 2                # 优化空间
        # dp[0] = 1
        # dp[1] = 1
        # for i in range(2, n + 1):
        #     tmp = sum(dp)
        #     dp[0] = dp[1]
        #     dp[1] = tmp
        # return dp[-1]

        # 或者可以这样优化空间
        if n == 1:
            return 1
        dp = [0] * 3
        dp[1] = 1
        dp[2] = 2
        for i in range(3, n + 1):
            tmp = dp[1] + dp[2]
            dp[1] = dp[2]
            dp[2] = tmp
        return dp[-1]


"""
动规五部曲：
    1.明确dp下标及dp[i]的含义
    2.确定递推公式
    3.初始化
    4.确定遍历顺序
    5.
"""