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
        n = len(nums)
        dp = [0 for _ in range(n + 1)]
        dp[1] = nums[0]
        for i in range(2, n + 1):
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i - 1])
        return dp[n]


if __name__ == "__main__":
    sl = Solution()

    cost = [2,7,9,3,1]
    print(sl.rob(cost))
