"""
12.25   圣诞快乐，那就从这里开始-->动态规划
    动态规划五部曲：
        1.确定dp数组及其下标i的含义
        2.确定递推公式
        3.初始化dp数组
        4.确定遍历顺序
        5.手动推导
"""
from typing import List, Optional

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

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
    
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        """ 494.目标和 \n 
            之前背包都是最多能装多少, 本题开始涉及装满有几种方式-->组合问题 """
        # left - right = target
        # left + right = sum,   由两个公式得出left = (target + sum) / 2, 题目变成了取出和为left的元素有几种方式

        # 一维dp: 既然是一维dp了, 初始化也就只要考虑行的初始化
        # 空间:O(sum(nums)) 时间:O(n * sum(nums))
        _sum = sum(nums)
        if abs(target) > _sum:
            return 0
        if (target + _sum) % 2:
            return 0
        left = (target + _sum) // 2
        dp = [0] * (left + 1)
        dp[0] = 1
        for i in range(0, len(nums)):       # 遍历物品必须从0开始了, 否则报错
            for j in range(left, nums[i] - 1, -1):
                dp[j] += dp[j - nums[i]]
        return dp[-1]
    
        # 一维dp: 和答案略有出入, 可以AC. 感觉答案更简单些
        _sum = sum(nums)
        if abs(target) > _sum:
            return 0
        if (target + _sum) % 2:
            return 0
        left = (target + _sum) // 2
        dp = [0] * (left + 1)           # 装满容量为j的背包有多少种方式
        # 初始化(其实也要考虑二维dp的首行首列初始化)
        if nums[0] <= left:
            dp[nums[0]] = 1
        dp[0] = 1
        if nums[0] == 0:
            dp[0] = 2
        for i in range(1, len(nums)):
            for j in range(left, nums[i] - 1, -1):
                dp[j] += dp[j - nums[i]]
        return dp[-1]

        # 空间:O(len(nums) * sum(nums)) 时间:O(len(nums) * sum(nums))
        _sum = sum(nums)
        if abs(target) > _sum:      # 必须判断, 否则有case无法通过
            return 0
        if (target + _sum) % 2:
            return 0
        left = (target + _sum) // 2
        dp = [[0] * (left + 1) for _ in range(len(nums))]   # 从0-i物品中挑选物品装满容量为j的背包有dp[i][j]种方式

        # 初始化首行
        if nums[0] <= left:
            dp[0][nums[0]] = 1
        dp[0][0] = 1        # 保证dp[0][0]为1. 这个初始化必须放到这里, 因为首列初始化可能会覆盖更新
        # 初始化首列
        zero_num = 0
        for i in range(len(nums)):
            if nums[i] == 0:
                zero_num += 1
            dp[i][0] = 2 ** zero_num

        # 开始遍历, 二维dp 先遍历物品或背包均可
        for i in range(1, len(nums)):
            for j in range(0, left + 1):        # 遍历背包容量从0或1开始都可以, 试过了
                if j >= nums[i]:
                    dp[i][j] = dp[i - 1][j] + dp[i - 1][j - nums[i]]        # 不放nums[i] + 放nums[i]
                elif j < nums[i]:
                    dp[i][j] = dp[i - 1][j]                                 # 放不下nums[i]
        return dp[-1][-1]

    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        """ 474.一和零 \n
            01背包 一维dp """
        from collections import Counter

        # 空间:O(m * n) 时间:O(len(s) * m * n)
        dp = [[0] * (n + 1) for _ in range(m + 1)]        # strs中最多有 i个0 j个1 的子集大小为dp[i][j]. i, j即背包的两个维度
        for s in strs:          # 先遍历物品
            zero_num = Counter(s)['0']
            one_num  = Counter(s)['1']
            for i in range(m, zero_num - 1, -1):
                for j in range(n, one_num - 1, -1):
                    dp[i][j] = max(dp[i][j], dp[i - zero_num][j - one_num] + 1)
        return dp[-1][-1]
    
    def complete_bag(self):
        """ 完全背包_理论基础 \n
            完全背包与01背包唯一区别: 遍历背包顺序 """
        weights = [1, 3, 4]
        values = [15, 20, 30]
        bagweight = 4

        dp = [0] * (bagweight + 1)      # 容量为j的背包所能装下物品的最大价值为dp[j]
        for i in range(0, len(weights)):
            for j in range(weights[i], bagweight + 1):  # 完全背包, 遍历背包必须正序
                dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
        return dp[-1]

    def change(self, amount: int, coins: List[int]) -> int:
        """ 518.零钱兑换II \n
            组合没有顺序, 排列才有顺序 """
        # 本题思路: 完全背包
        #          组合问题
        # 空间:O(n) 时间:O(n * m)
        dp = [0] * (amount + 1)         # 装满容量为j的背包的硬币组合数为dp[j]
        dp[0] = 1
        for i in range(len(coins)):
            for j in range(coins[i], amount + 1):
                dp[j] += dp[j - coins[i]]
        return dp[-1]
    
    def combinationSum4(self, nums: List[int], target: int) -> int:
        """ 377.组合总和IV """
        # 完全背包: 1.物品数量无限 2.先遍历物品 再遍历背包, 内层遍历顺序正序
        # 组合问题: 递推公式dp[j] += dp[j - nums[i]]
        #   排列问题: 那for循环就有讲究了, 先遍历背包 再遍历容量. 因为: 若外层为遍历物品则只能得到[1, 3], 不会得到[3, 1], 而排列是有顺序的
        # 空间:O(target) 时间:O(target * len(nums))
        dp = [0] * (target + 1)         # 装满容量为j的背包有dp[j]种组合方式
        dp[0] = 1
        for i in range(target + 1):
            for j in range(len(nums)):
                if nums[j] <= i:
                    dp[i] += dp[i - nums[j]]
        return dp[-1]
    
    def climb_(self, n, m):
        """ 爬楼梯(进阶版) \n
            一共n个台阶, 每次至多爬m个台阶 """
        # 完全背包
        # 排列问题
        dp = [0] * (n + 1)
        dp[0] = 1
        for i in range(n + 1):          # 先遍历背包
            for j in range(1, m + 1):   # 再遍历物品
                if j <= i:
                    dp[i] += dp[i - j]
        return dp[-1]

    def coinChange(self, coins: List[int], amount: int) -> int:
        """ 322.零钱兑换 """
        # 完全背包: 
        #       1.外层遍历物品, 内层遍历背包
        #       2.内层遍历正序, 因为可重复放入
        # 组合问题: 
        #       1.递推公式 dp[j] += dp[j - nums[i]]
        # 空间:O(len(amount)) 时间:O(len(coins) * amount)
        dp = [float('inf')] * (amount + 1)         # 能凑成金额j的最少硬币个数为dp[j]
        dp[0] = 0
        for i in range(len(coins)):
            for j in range(coins[i], amount + 1):
                dp[j] = min(dp[j], dp[j - coins[i]] + 1)
        return dp[-1] if dp[-1] != float('inf') else -1

    def numSquares(self, n: int) -> int:
        """ 279.完全平方数 """
        # 完全背包--物品数量不限
        # 组合问题
        # 时间复杂度:O(n * sqrt(n)) 空间复杂度:O(n)
        dp = [float('inf')] * (n + 1)      # 装满容量为j的背包需要完全平方数的最少个数dp[j]
        dp[0] = 0                          # 纯为推导 没意义(自己做的时候能想到吗)
        for i in range(1, n + 1):               # 遍历物品
            for j in range(i ** 2, n + 1):      # 遍历背包
                dp[j] = min(dp[j], dp[j - i ** 2] + 1)
        return dp[-1]

    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        """ 139.单词拆分 """
        # 完全背包--物品数量不限:
        #               1.外层遍历物品, 内层遍历背包
        #               2.内层遍历可以正序, 因为可重复放入
        # 排列问题--要顺序:
        #               1.外层遍历背包, 内层遍历物品 (若外层遍历物品, 只能得到[1, 3], 永远得不到[3, 1]这种排列)
        #               2.递推公式dp[j] += dp[j - nums[i]]
        # 单词拆分的递推公式需要认真想一下哈, 不要生搬硬套
        # 时间复杂度:O(n ^ 2) 空间复杂度:O(n)
        dp = [False] * (len(s) + 1)         # 一定要区分dp与s的索引, 二者的意义不同. 前j个字母能否用wordDict来表示/容量为j的背包是否能装满
        dp[0] = True            # 纯为递推
        for i in range(1, len(s) + 1):      # 遍历背包
            for j in range(i):              # 遍历物品
                if s[j:i] in wordDict and dp[j]:    # 若前j个字母能被wordDict表示, (s[j:i]表示前j个后续的一段)且s[j:i]又在wordDict中, 即前i个字母能被表示, dp[i] = True
                    dp[i] = True
        return dp[-1]
    
    def rob(self, nums: List[int]) -> int:
        """ 198.打家劫舍 """
        # 时间复杂度:O(n) 空间复杂度:O(n)
        n = len(nums)
        if n == 1:
            return nums[0]
        dp = [0] * n
        dp[0] = nums[0]
        dp[1] = max(nums[:2])
        for i in range(2, n):
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
        return dp[-1]
    
    def robII(self, nums: List[int]) -> int:
        """ 231.打家劫舍II \n
            成环 """
        # 时间复杂度:O(n) 空间复杂度:O(n)
        def rob(nums, start, end):
            """ 198.打家劫舍 """
            print(f'---- 调用此rob')
            nums = nums[start:end + 1]
            n = len(nums)
            if n == 1:
                return nums[0]
            dp = [0] * n
            dp[0] = nums[0]
            dp[1] = max(nums[:2])
            for i in range(2, n):
                dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
            return dp[-1]
        
        n = len(nums)
        if n == 1:
            return nums[0]
        res1 = rob(nums, 0, n - 2)
        res2 = rob(nums, 1, n - 1)
        return max(res1, res2)
    
    def robIII(self, root: Optional[TreeNode]) -> int:
        """ 337.打家劫舍III """
        # 要复习二叉树!
        # 时间复杂度:O(n) 空间复杂度:O(logn)
        def rob_(root):
            if not root:
                return [0, 0]
            left =  rob_(root.left)
            right = rob_(root.right)
            return [max(left) + max(right), root.val + left[0] + right[0]]

        res = rob_(root)
        return max(res)
    
    def maxProfit(self, prices: List[int]) -> int:
        """ 121.买卖股票的最佳时机 \n
            买卖一次 """
        # 时间:O(n) 空间:O(n)
        n = len(prices)
        dp = [[0] * 2 for _ in range(n)]        # dp[i][0]表示持有 dp[i][1]表示不持有
        dp[0] = [-prices[0], 0]
        for i in range(1, n):
            dp[i][0] = max(dp[i - 1][0], -prices[i])
            dp[i][1] = max(dp[i - 1][1], prices[i] + dp[i - 1][0])
        return max(dp[-1])

if __name__ == '__main__':
    sl = Solution()

    prices = [7,6,4,3,1]
    print(sl.maxProfit(prices))
