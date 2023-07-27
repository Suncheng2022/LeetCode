from typing import List


class Solution:
    def numTilings(self, n: int) -> int:
        """ 790.多米诺和托米诺平铺 """
        dp = [[0] * 4 for _ in range(n + 1)]
        dp[0][3] = 1    # i从1开始计数，0表示一列都没有，即填满；或者就硬记住
        MOD = 10 ** 9 + 7
        for i in range(1, n + 1):
            dp[i][0] = dp[i - 1][3]
            dp[i][1] = dp[i - 1][0] + dp[i - 1][2]
            dp[i][2] = dp[i - 1][0] + dp[i - 1][1]
            dp[i][3] = dp[i - 1][0] + dp[i - 1][1] + dp[i - 1][2] + dp[i - 1][3]
        return dp[-1][-1] % MOD

    def mincostTickets(self, days: List[int], costs: List[int]) -> int:
        """ 983.最低票价 """
        # 答案 需要用到递归完成回溯
        memo = {}
        days = set(days)

        def func(i):    # 计算dp[i]   第i天到今年结束所需花费最少费用
            if i in memo:
                return memo[i]

            if i > 365:
                return 0
            elif i in days:
                memo[i] = min(func(i + d) + c for d, c in zip([1, 7, 30], costs))
                return memo[i]
            else:
                return func(i + 1)

        return func(1)

    def numDecodings(self, s: str) -> int:
        """ 91.解码方法 """
        n = len(s)
        dp = [0] * (n + 1)      # dp[i] 截止到前i位，解码方式有多少种
        dp[0] = 1       # 前0位是空，能解码出空字符串这1种解码方式
        for i in range(1, n + 1):   # 计算dp[i]  当前处理的是s[i]
            if s[i - 1] != '0':     # 上一次 解码1位，则截止到这一位的解码方式和截止到上一位的解码方式数量相同
                dp[i] = dp[i - 1]
            if i > 1 and s[i - 2] != '0' and int(s[i - 2:i]) <= 26:     # 上一次 解码2位
                dp[i] += dp[i - 2]
        return dp[-1]




if __name__ == '__main__':
    sl = Solution()

    s = '06'
    print(sl.numDecodings(s))
