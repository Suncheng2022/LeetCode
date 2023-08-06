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

    def countGoodStrings(self, low: int, high: int, zero: int, one: int) -> int:
        """ 2466.统计构造好字符串的方案数 """
        dp = [0] * (high + 1)       # dp[i]表示 构造长度为i的字符串的数目/方式
        dp[0] = 1       # 构造长度0的字符串有1种构造方式；通常动态规划总要初始化开头为1
        for i in range(1, high + 1):    # 根据上一次dp来更新dp[i]，注意这里是累加，因为2种方式都符合要求
            if i >= zero:
                dp[i] += dp[i - zero]
            if i >= one:
                dp[i] += dp[i - one]
        return sum(dp[low:]) % (10 ** 9 + 7)

    def coinChange(self, coins: List[int], amount: int) -> int:
        """ 322.零钱兑换
            题目求的是组成amount所需 最少硬币个数，没有要求顺序 """
        dp = [float('inf')] * (amount + 1)     # dp[i]表示 组成金额i 所需最少硬币个数；注意初始化，初始化为最大的数了，否则结果返回0、-1时不好处理
        dp[0] = 0       # 组成金额0的硬币数
        for i in range(1, amount + 1):
            for coin in coins:
                if i >= coin:
                    dp[i] = min(dp[i], dp[i - coin] + 1)
        return dp[-1] if dp[-1] != float('inf') else -1

    def mostPoints(self, questions: List[List[int]]) -> int:
        """ 2140.解决智力问题
            做or不做 当前题会影响后面的能做的题目，考虑无后效性，使用反向动态规划 """
        n = len(questions)
        dp = [0] * (n + 1)      # dp[i]表示做当前题及以后所有题(不一定都做，是考虑范围) 获得的最高得分；dp[n]特殊处理
        for i in range(n - 1, -1, -1):      # dp[n]特殊处理，就当没做题看待  dp[0]=0
            # 对每道题，有做/不做2种选择，我们选择得分较大的方案
            dp[i] = max(dp[i + 1],      # 不做当前题
                        # 做当前题，则后面若干道不能做了，取能做的那些 所能得到的最高分数; min, 若索引超出范围 使用min给拉回来
                        questions[i][0] + dp[min(n, i + questions[i][1] + 1)])
        return dp[0]

    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        """ 474.一和零
            背包问题：物品和容量，三维背包问题，考虑两个容量 """
        from collections import Counter

        dp = [[[0] * (n + 1) for _ in range(m + 1)] for _ in range(len(strs) + 1)]      # dp[i][j][k]
        for i in range(1, len(strs) + 1):       # i=0 即前0个字符串结果肯定都为0，所以从1开始遍历
            # 统计当前字符串'0'和'1'的数量
            zeros = Counter(strs[i - 1])['0']
            ones = Counter(strs[i - 1])['1']
            # 针对前i个字符串，动态规划 计算包含m个0、n个1的最大子集长度
            for j in range(m + 1):
                for k in range(n + 1):
                    if j < zeros or k < ones:       # 如果j、k小于当前字符串0、1的数目，则当前字符串不能选，超出j、k数量限制了
                        dp[i][j][k] = dp[i - 1][j][k]
                    elif j >= zeros and k >= ones:  # 可以选当前字符串，则 选
                        dp[i][j][k] = max(dp[i - 1][j][k], dp[i - 1][j - zeros][k - ones] + 1)
        return dp[-1][-1][-1]

    def combinationSum4(self, nums: List[int], target: int) -> int:
        """ 377.组合总和IV
            dp[i] 和为i的组合数 """
        dp = [0] * (target + 1)
        dp[0] = 1       # 和为0只有1种方案--什么都不选
        for i in range(1, target + 1):
            # 计算每个dp[i]均遍历nums
            for n in nums:
                if n <= i:
                    dp[i] += dp[i - n]      # 和为i-n 的组合后面加上 元素n 就是 和i
        return dp[-1]


if __name__ == '__main__':
    sl = Solution()

    nums = [1,2,3]
    target = 4
    print(sl.combinationSum4(nums, target))
