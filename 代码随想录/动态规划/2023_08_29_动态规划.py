from typing import List


class Solution:
    def fib(self, n: int) -> int:
        """ 509.斐波那契数 """
        if n <= 1:
            return n
        dp = [0] * (n + 1)
        dp[0], dp[1] = 0, 1
        for i in range(2, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[n]

    def climbStairs(self, n: int) -> int:
        """ 70.爬楼梯 """
        if n <= 1:
            return n
        dp = [0] * (n + 1)
        dp[1], dp[2] = 1, 2     # 爬到索引i楼梯，有dp[i]种方法; 不使用dp[0]，因为dp[0]如何初始化原因解释不通
        for i in range(3, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[n]

        # 优化一下空间复杂度
        # if n <= 1:
        #     return n
        # dp = [1, 2]
        # for i in range(3, n + 1):
        #     sum = dp[0] + dp[1]
        #     dp[0] = dp[1]
        #     dp[1] = sum
        # return dp[1]

    def minCostClimbingStairs(self, cost: List[int]) -> int:
        """ 746.使用最小花费爬楼梯 """
        dp = [0] * (len(cost) + 1)      # dp的含义要明确，到达下标i台阶所需要的最小花费为dp[i]
        dp[0], dp[1] = 0, 0     # 题目明确说了，可以选择从下标为0或1的台阶开始爬，即到达下标0或1的台阶不花费，继续往上爬就要花费体力
        for i in range(2, len(cost) + 1):
            # 状态转移公式：dp[i-1]转移到dp[i]——即从下标i-1继续往上爬就需要花费体力了，体力就是cost[i-1]；下标i-2往上爬类似
            dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])
        return dp[-1]

        # 优化 空间复杂度
        # dp = [0] * 2
        # dp[0], dp[1] = 0, 0     # 起点是下标0或1，到达起点不花费，继续往上爬才需要花费体力
        # for i in range(2, len(cost) + 1):
        #     dp_i = min(dp[0] + cost[i - 2], dp[1] + cost[i - 1])    # 有点绕，想一下索引i=2如何得到吧
        #     dp[0] = dp[1]
        #     dp[1] = dp_i
        # return dp[1]



if __name__ == '__main__':
    sl = Solution()

    cost = [10,15,20]
    print(sl.minCostClimbingStairs(cost))

    """
    动态规划五部曲：
        1.确定dp数组以及下标的含义
        2.确定递推公式
        3.dp数组如何初始化
        4.确定遍历顺序
        5.举例推导dp数组
    """