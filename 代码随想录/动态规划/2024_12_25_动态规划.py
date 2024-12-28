"""
12.25   圣诞快乐，那就从这里开始-->动态规划
    动态规划五部曲：
        1.确定dp数组及其下标i的含义
        2.确定递推公式
        3.初始化dp数组
        4.确定遍历顺序
        5.手动推导
"""
from typing import List


class Solution:
    def fib(self, n: int) -> int:
        # 空间复杂度:O(1) 时间复杂度:O(n)
        if n <= 1:
            return n

        dp = [0] * 2        # 降低空间复杂度
        dp[0] = 0
        dp[1] = 1
        for i in range(2, n + 1):
            _sum = dp[0] + dp[1]
            dp[0], dp[1] = dp[1], _sum
        return dp[-1]

        # 空间复杂度:O(n) 时间复杂度:O(n)
        # if n <= 1:
        #     return n
        #
        # dp = [0] * (n + 1)      # 第i个费锲那波数是dp[i]
        # dp[0] = 0
        # dp[1] = 1
        # for i in range(2, n + 1):
        #     dp[i] = dp[i - 1] + dp[i - 2]
        # return dp[n]

    def climbStairs(self, n: int) -> int:
        # 空间复杂度:O(1) 时间复杂度:O(n)
        if n <= 2:
            return n
        dp = [0] * 3
        dp[1] = 1
        dp[2] = 2
        for i in range(3, n + 1):
            _sum = dp[1] + dp[2]
            dp[1], dp[2] = dp[2], _sum
        return dp[-1]

        # 空间复杂度:O(n) 时间复杂度:O(n)
        # if n <= 2:
        #     return n
        #
        # dp = [0] * (n + 1)      # 爬到第i层楼梯, 有dp[i]种方法
        # dp[1] = 1
        # dp[2] = 2
        # for i in range(3, n + 1):
        #     dp[i] = dp[i - 1] + dp[i - 2]
        # return dp[n]

    def minCostClimbingStairs(self, cost: List[int]) -> int:
        """ 746. 使用最小花费爬楼梯 """
        # 重做一遍
        # 空间复杂度:O(1) 时间复杂度:O(n)
        n = len(cost)
        dp = [0] * 2        # 优化空间复杂度, dp[i]由前两项推导来, 只保留前两项就可以
        dp[0] = 0
        dp[1] = 0
        for i in range(2, n + 1):
            _tmp = min(dp[1] + cost[i - 1], dp[0] + cost[i - 2])
            dp[0], dp[1] = dp[1], _tmp
        return dp[-1]

        # 空间复杂度:O(n) 时间复杂度:O(n)
        # n = len(cost)       # 一共有n个台阶
        # dp = [0] * (n + 1)  # 到达索引第i个台阶, 最小花费为dp[i]. 注意: 题目让求的是到达楼梯顶部, 而不是到达第n个台阶--即要越过所有台阶, 才叫到达楼梯顶部
        # dp[0] = 0
        # dp[1] = 0
        # for i in range(2, n + 1):
        #     dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])
        # return dp[-1]

        # 一时想不起来思路, 看下之前自己做题记录

        # 空间复杂度:O(1) 时间复杂度:O(n)
        # n = len(cost)
        # dp = [0] * 2        # 优化空间复杂度
        # dp[0] = 0
        # dp[1] = 0
        # for i in range(2, n + 1):
        #     _tmp = min(dp[1] + cost[i - 1], dp[0] + cost[i - 2])
        #     dp[0], dp[1] = dp[1], _tmp
        # return dp[-1]

        # 空间复杂度:O(n) 时间复杂度:O(n)
        # n = len(cost)           # 一共有n个台阶
        # dp = [0] * (n + 1)      # 到达索引第i个台阶花费最小费用为dp[i]. 将n个台阶全都跳过去, 才叫到达楼梯顶部
        # dp[0] = 0
        # dp[1] = 0
        # for i in range(2, n + 1):
        #     dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])
        # return dp[-1]

    def uniquePaths(self, m: int, n: int) -> int:
        """ 62.不同路径 """
        # 尝试优化空间复杂度
        # 空间复杂度:O(n) 时间复杂度:O(m*n)
        dp = [1] * n                    # 初始化第一行
        for i in range(1, m):
            for j in range(1, n):
                dp[j] += dp[j - 1]
        return dp[-1]

        # 空间复杂度:O(m * n) 时间复杂度:O(m * n)
        # dp = [[0] * n for _ in range(m)]        # mxn的二维数组. 从起点到达[i,j]有dp[i][j]种不同路径
        # for w in range(n):
        #     dp[0][w] = 1
        # for h in range(m):
        #     dp[h][0] = 1
        # for i in range(1, m):
        #     for j in range(1, n):
        #         dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        # return dp[-1][-1]

    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        """ 63.不同路径II """
        # 尝试优化空间复杂度, 有些复杂了就
        # 空间复杂度:O(n) 时间复杂度:O(m * n)
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        dp = [0] * n                            # dp只保留一行
        for i in range(n):
            if obstacleGrid[0][i]:
                break
            dp[i] = 1
        for i in range(1, m):
            for j in range(n):
                if obstacleGrid[i][j]:
                    dp[j] = 0
                elif j != 0:
                    dp[j] += dp[j - 1]
        return dp[-1]

        # 空间复杂度:O(m * n) 时间复杂度:O(m * n)
        # m = len(obstacleGrid)
        # n = len(obstacleGrid[0])
        # dp = [[0] * n for _ in range(m)]        # 到达[i,j]有dp[i][j]条不同路径
        # for w in range(n):
        #     if obstacleGrid[0][w]: break
        #     dp[0][w] = 1
        # for h in range(m):
        #     if obstacleGrid[h][0]: break
        #     dp[h][0] = 1
        # for i in range(1, m):
        #     for j in range(1, n):
        #         if obstacleGrid[i][j]:
        #             continue
        #         dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        # return dp[-1][-1]

    def integerBreak(self, n: int) -> int:
        """ 343.整数拆分 """
        # 拆分为m个最相似的数, 乘积才会最大, 只不过m不知道而已, 但至少>=2
        # 空间复杂度:O(n) 时间复杂度:O(n^2)
        # dp = [0] * (n + 1)
        # dp[2] = 1
        # for i in range(3, n + 1):
        #     for j in range(1, i // 2 + 1):
        #         dp[i] = max(dp[i], max(j * (i - j), j * dp[i - j]))
        # return dp[-1]

        # 空间复杂度:O(n) 时间复杂度:O(n^2)
        # dp = [0] * (n + 1)          # 拆分i的最大乘积为dp[i]
        # dp[2] = 1                   # dp[0] dp[1]就没有意义, 不用管
        # for i in range(3, n + 1):
        #     for j in range(1, i):
        #         dp[i] = max(dp[i], max(j * (i - j), j * dp[i - j]))     # 因为内层for遍历的时候在不断更新dp[i], 要保留dp[i]的最大值
        # return dp[-1]

        # 再做一遍
        dp = [0] * (n + 1)      # 拆分i得到的最大乘积dp[i]
        dp[2] = 1
        for i in range(3, n + 1):
            for j in range(1, i // 2 + 1):
                dp[i] = max(dp[i], max(j * (i - j), j * dp[i - j]))
        return dp[-1]

    def numTrees(self, n: int) -> int:
        """ 96.不同的二叉搜索树 """
        # 空间: O(n) 时间:O(n^2)
        dp = [0] * (n + 1)          # i个数能组成dp[i]个不同的二叉搜索树. 为什么n+1个元素呢--dp[n]是题目所求, 索引得能到n
        dp[0] = 1
        for i in range(1, n + 1):
            for j in range(1, i + 1):
                dp[i] += dp[j - 1] * dp[i - j]
        return dp[-1]

    def bag01(self):
        """ 01背包 基础理论_一 """
        weights = [1, 3, 4]
        values = [15, 20, 30]
        bagweight = 4

        dp = [[0] * (bagweight + 1) for _ in range(len(weights))]       # dp[i][j] 从0-i物品中, 放入容量为j的背包中, 能得到最大的价值是dp[i][j]
        for j in range(weights[0], bagweight + 1):                      # 初始化 第一列dp[i][0]为0, 第一行dp[0][j]中j>=weights[0]的初始化为weights[0]
            dp[0][j] = values[0]
        for i in range(1, len(weights)):
            for j in range(1, bagweight + 1):
                if j < weights[i]:
                    dp[i][j] = dp[i - 1][j]
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weights[i]] + values[i])
        # print(f'$$$$$$$')
        # for line in dp:
        #     print(line)
        return dp[-1][-1]

    def bag01_1d(self):
        """ 01背包 基础理论_二
            将dp压缩为一维数组. 推荐写法, 空间复杂还降了一个数量级 """
        weights = [1, 3, 4]
        values = [15, 20, 30]
        bagweight = 4

        dp = [0] * (bagweight + 1)      # 背包容量为j能放下物品的最大价值为dp[j]
        for i in range(len(weights)):                       # 注意:一维背包, 遍历顺序只能先物品再背包
            for j in range(bagweight, weights[i] - 1, -1):  # 注意:一维背包, 背包遍历顺序只能倒序, 防止覆盖
                dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
        return dp[-1]
    
    def canPartition(self, nums: List[int]) -> bool:
        """ 416.分割等和子集 """
        # 空间:O(n) 时间:O(n ^ 2)
        if sum(nums) % 2:
            return False
        
        target = sum(nums) // 2
        dp = [0] * (target + 1)     # 容量为j的背包, 能装下最大物品价值为dp[j]
        for i in range(len(nums)):
            for j in range(target, nums[i] - 1, -1):
                dp[j] = max(dp[j], dp[j - nums[i]] + nums[i])
        return max(dp) == target
    
    def lastStoneWeightII(self, stones: List[int]) -> int:
        """ 1049.最后一块石头的重量II """
        # 空间:O(石头总重量) 时间:O(石头总重量 * 石头数量)
        target = sum(stones) // 2
        dp = [0] * (target + 1)     # 容量为j的背包, 能装下最大物品价值为dp[i]
        for i in range(len(stones)):
            for j in range(target, stones[i] - 1, -1):
                dp[j] = max(dp[j], dp[j - stones[i]] + stones[i])
        return sum(stones) - dp[-1] * 2

if __name__ == '__main__':
    sl = Solution()

    stones = [31,26,33,21,40]
    print(sl.lastStoneWeightII(stones))
