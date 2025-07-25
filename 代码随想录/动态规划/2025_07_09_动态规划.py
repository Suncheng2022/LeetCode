"""
07.09   是的, 该走了
"""
from typing import List, Optional
from ipdb import set_trace as st

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def fib(self, n: int) -> int:
        """ 509.斐切那波数 """
        # 时间:O(n) 空间:O(1)
        # if n <= 1:
        #     return n
        # dp = [0, 1]     # 保留两个结果
        # for i in range(2, n + 1):
        #     tmp = sum(dp)
        #     dp[0], dp[1] = dp[1], tmp
        # return dp[-1]

        # 递归
        # 时间:O(2^n) 二叉树节点数 空间:O(n) 栈的最大深度就是最长调用链
        if n <= 1:
            return n
        return self.fib(n - 1) + self.fib(n - 2)

        # 时间:O(n) 空间:O(n) 
        # if n <= 1:
        #     return n
        # dp = [0] * (n + 1)
        # dp[0], dp[1] = 0, 1
        # for i in range(2, n + 1):
        #     dp[i] = dp[i - 1] + dp[i - 2]
        # return dp[n]

    def climbStairs(self, n: int) -> int:
        """ 70.爬楼梯 """
        # 时间:O(n) 空间:O(1)
        if n <= 2:
            return n
        dp = [0] * 3            # dp[0]无意义
        dp[1], dp[2] = 1, 2     # 初始化
        for i in range(3, n + 1):
            tmp = dp[1] + dp[2]
            dp[1], dp[2] = dp[2], tmp
        return dp[-1]

        # 时间:O(n) 空间:O(n)
        # if n <= 2:
        #     return n
        # dp = [0] * (n + 1)      # dp[i]表示爬到第i个台阶有dp[i]种方法
        # dp[1] = 1
        # dp[2] = 2
        # for i in range(3, n + 1):
        #     dp[i] = dp[i - 1] + dp[i - 2]
        # return dp[-1]

    def minCostClimbingStairs(self, cost: List[int]) -> int:
        """ 746.使用最小花费爬楼梯 \n
            理解好题意, 可以从台阶0或1开始, 就是到达台阶0或1是不花钱的\n
                        最终的目的是越过所有台阶, 所以dp长度为n+1 \n
            可以参考下<代码随想录>本题推导步骤的图片 """
        # 时间:O(n) 空间:O(1)
        n = len(cost)
        dp = [0, 0]     # 优化空间, dp[i]通过前两项推导得来, 所以保留最新的两项即可
        for i in range(2, n + 1):
            _min = min(dp[-1] + cost[i - 1], dp[-2] + cost[i - 2])
            dp[0], dp[1] = dp[1], _min
        return dp[-1]

        # 时间:O(n) 空间:O(n)
        # n = len(cost)
        # if n <= 2:
        #     return min(cost)              # 最终目的是越过所有台阶, 即便只有2个台阶, 也要越呀
        # dp = [0] * (n + 1)                # dp[i]表示到达下标为i的台阶所需最小花费
        # for i in range(2, n + 1):
        #     dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])
        # return dp[-1]

    def uniquePaths(self, m: int, n: int) -> int:
        """ 62.不同路径 """
        # 时间:O(m*n) 空间:O(m*n)
        # if m == n == 1:
        #     return 1
        # dp = [[0] * n for _ in range(m)]    # dp[i][j]表示到达第i行第j列有dp[i][j]个不同路径
        # for i in range(m):
        #     dp[i][0] = 1
        # for j in range(n):
        #     dp[0][j] = 1
        # for i in range(1, m):
        #     for j in range(1, n):
        #         dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        # return dp[-1][-1]
        
        # 时间:O(m*n) 空间:O(n)
        dp = [1] * n        # 滚动数组
        for i in range(1, m):
            for j in range(1, n):
                dp[j] += dp[j - 1]
        return dp[-1]
    
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        """ 63.不同路径II """
        # 时间:O(m*n) 空间:O(m*n)
        # m, n = len(obstacleGrid), len(obstacleGrid[0])
        # dp = [[0] * n for _ in range(m)]        # dp[i][j]表示到达位置[i,j]有多少种不同方法
        # for i in range(m):
        #     if obstacleGrid[i][0] == 1:
        #         break
        #     dp[i][0] = 1
        # for j in range(n):
        #     if obstacleGrid[0][j] == 1:
        #         break
        #     dp[0][j] = 1
        # for i in range(1, m):
        #     for j in range(1, n):
        #         if obstacleGrid[i][j] == 1:
        #             continue
        #         dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        # return dp[-1][-1]

        # 时间:O(mxn) 空间:O(n)
        # 优化空间有点不好想, 主要是要处理遍历的时候
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        dp = [0] * n    # 滚动数组 优化空间
        for i in range(n):
            if obstacleGrid[0][i] == 1:
                break
            dp[i] = 1
        for i in range(1, m):
            for j in range(n):
                if obstacleGrid[i][j] == 1:
                    dp[j] = 0
                elif j > 0:     # 如果不加>0的限制条件, 会有dp[0] += dp[-1], 对不
                    dp[j] += dp[j - 1]
        return dp[-1]

    def integerBreak(self, n: int) -> int:
        """ 343.整数拆分 """
        # 时间:O(n^2) 空间:O(n)
        dp = [0] * (n + 1)        # dp[i]表示拆分i所得最大乘积
        dp[2] = 1                 # dp[0] dp[1]就没有意义
        for i in range(3, n + 1):
            for j in range(1, i // 2 + 1):
                dp[i] = max(dp[i], max((i - j) * j, dp[i - j] * j))     # 拆分公式/推导公式不好想出来
        return dp[-1]
    
    def numTrees(self, n: int) -> int:
        """ 96.不同的二叉搜索树 """
        # 时间:O(n^2) 空间:O(n)
        dp = [0] * (n + 1)      # dp[i]表示 i个不同节点/从1到i共i个不同节点 所组成不同二叉搜索树的数量
        dp[0] = 1
        for i in range(1, n + 1):
            for j in range(1, i + 1):
                dp[i] += dp[j - 1] * dp[i - j]      # 同样, 递推公式不好想
        return dp[-1]
    
    def bag01(self):
        """ 01背包理论 """
        weights = [1, 3, 4]
        values = [15, 20, 30]
        bagWeight = 4
        dp = [[0] * (bagWeight + 1) for _ in range(len(weights))]   # dp[i][j] 从0~i选择物品, 背包容量为j, 最大价值为dp[i][j]
        for j in range(weights[0], bagWeight + 1):
            dp[0][j] = values[0]
        for i in range(1, len(weights)):        # 先遍历物品
            for j in range(weights[i], bagWeight + 1):  # 再遍历背包
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weights[i]] + values[i])
        return dp[-1][-1]
    
    def test_kama46(self):
        """ 卡码网 46. 携带研究材料 """
        m, n = map(int, input().split())        # m:物品数量 n:背包容量
        weights = list(map(int, input().split()[:m]))
        values = list(map(int, input().split()[:m]))
        # m, n = 6, 1
        # weights = [2, 2, 3, 1, 5, 2]
        # values =  [2, 3, 1, 5, 4, 3]
        # _wv = [[w, v] for w, v in zip(weights, values)]       # 下面遍历容量的地方从头开始后, 这里也不需要了
        # _wv.sort(key=lambda x: x[0])
        # weights.clear()
        # values.clear()
        # for w, v in _wv:
        #     weights.append(w)
        #     values.append(v)
        # print(f'>>> m:{m} n:{n}\n'
        #       f'>>> weights:{weights}\n'
        #       f'>>> values: {values}')

        dp = [[0] * (n + 1) for _ in range(m)]  # dp[i][j] 从0~i中选取物品 背包容量为j, 最大价值是dp[i][j]
        for j in range(weights[0], n + 1):
            dp[0][j] = values[0]

        for i in range(1, m):
            for j in range(1, n + 1):       # 遍历的起始位置, 保险起见从头开始, 有可能会报错
                if j < weights[i]:
                    dp[i][j] = dp[i - 1][j]
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weights[i]] + values[i])
        # for line in dp:
        #     print(line)
        print(dp[-1][-1])

    def bag01_(self):
        """ 01背包理论\n
            优化空间 滚动数组 """
        # m, n = 6, 1
        # weights = [2, 2, 3, 1, 5, 2]
        # values =  [2, 3, 1, 5, 4, 3]
        m, n = map(int, input().split())
        weights = list(map(int, input().split()))
        values = list(map(int, input().split()))
        dp = [0] * (n + 1)
        # for j in range(weights[0], n + 1):            # 不必初始化
        #     dp[j] = values[0]
        for i in range(m):
            for j in range(n, weights[i] - 1, -1):      # 01背包的一维数组一定倒序 否则会重复放入--原因由推导公式可知
                dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
        print(dp[-1])

    def canPartition(self, nums: List[int]) -> bool:
        """ 416.分割等和子集 """
        # 时间:O(mxn) 空间:O(n)
        if sum(nums) % 2:
            return False
        
        n = len(nums)
        bagWeight = sum(nums) // 2
        dp = [0] * (bagWeight + 1)      # dp[j]表示装满容量j的背包最大价值是dp[j]
        for i in range(n):
            for j in range(bagWeight, nums[i] - 1, -1):     # 一维dp, 遍历容量倒序
                dp[j] = max(dp[j], dp[j - nums[i]] + nums[i])
        # for line in dp:
        #     print(line)
        return dp[-1] == bagWeight
    
    def lastStoneWeightII(self, stones: List[int]) -> int:
        """ 1049.最后一块石头的重量II \n
            思路 416.分割等和子集 """
        # 时间:O(mxn) 空间:O(n)
        bagWeight = sum(stones) // 2
        dp = [0] * (bagWeight + 1)      # dp[j]表示装满容量为j的背包所得最大价值
        for i in range(len(stones)):
            for j in range(bagWeight, stones[i] - 1, -1):
                dp[j] = max(dp[j], dp[j - stones[i]] + stones[i])
        return sum(stones) - dp[-1] * 2
    
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        """ 494.目标和 \n
            由 left + right = sum \n
               left - right = target, 得left = (sum + target) / 2 \n
            之前是求容量为j的背包最多能装多少, 本题是求装满最多有多少种方法--组合问题 """
        # 时间:O(mxn) 空间:O(mxn)
        # if sum(nums) < abs(target):
        #     return 0
        # if (sum(nums) + target) % 2:
        #     return 0
        # m = len(nums)                       # 物品数量
        # n = (sum(nums) + target) // 2       # bagWeight
        # dp = [[0] * (n + 1) for _ in range(m)]  # dp[i][j] 从[0,i]选物品 装满 容量为j的背包, 最多有dp[i][j]种办法
        # _zeroNum = 0
        # for i in range(m):            # 初始化列, 注意nums元素为0
        #     if nums[i] == 0:
        #         _zeroNum += 1
        #     dp[i][0] = 2 ** _zeroNum
        # for j in range(1, n + 1):     # 初始化行, 这里没想清楚
        #     if j == nums[0]:
        #         dp[0][j] = 1
        # for i in range(1, m):
        #     for j in range(1, n + 1):
        #         if j < nums[i]:
        #             dp[i][j] = dp[i - 1][j]
        #         else:
        #             dp[i][j] = dp[i - 1][j] + dp[i - 1][j - nums[i]]
        # for line in dp:
        #     print(line)
        # return dp[-1][-1]

        ## 一维dp
        # 时间:O(mxn) 空间:O(n)
        # if (sum(nums) + target) % 2:
        #     return 0
        # if sum(nums) < abs(target):
        #     return 0
        # bagWeight = (sum(nums) + target) // 2
        # dp = [0] * (bagWeight + 1)      # dp[j] 装满容量为j的背包最多有dp[j]种方法
        # dp[0] = 1
        # for i in range(len(nums)):
        #     for j in range(bagWeight, nums[i] - 1, -1):
        #         dp[j] += dp[j - nums[i]]
        # return dp[-1]

        ## Again
        # left = (sum + target) / 2
        # 装满背包有多少种方法 
        ## 二维dp 推导公式dp[i][j] = dp[i - 1][j] + dp[i - 1][j - nums[i]]
        # if sum(nums) < abs(target):
        #     return 0
        # if (sum(nums) + target) % 2:
        #     return 0
        # bagWeight = (sum(nums) + target) // 2
        # dp = [[0] * (bagWeight + 1) for _ in range(len(nums))]  # dp[i][j] 从[0, i]选取物品装满容量为j的背包有dp[i][j]种方法
        # _zeroNum = 0
        # for i in range(len(nums)):          # 初始化首列
        #     if nums[i] == 0:
        #         _zeroNum += 1
        #     dp[i][0] = 2 ** _zeroNum
        # # if nums[0] <= bagWeight:            # 初始化首行 | 果然有问题, 会覆盖初始化的首列
        # #     dp[0][nums[0]] = 1
        # for j in range(1, bagWeight + 1):   # 初始化首行
        #     if j == nums[0]:
        #         dp[0][j] = 1
        # for i in range(1, len(nums)):
        #     for j in range(1, bagWeight + 1):
        #         if j < nums[i]:
        #             dp[i][j] = dp[i - 1][j]
        #         else:
        #             dp[i][j] = dp[i - 1][j] + dp[i - 1][j - nums[i]]
        # return dp[-1][-1]
        
        ## 一维dp 推导公式dp[j] += dp[j - nums[i]]
        if sum(nums) < abs(target):
            return 0
        if (sum(nums) + target) % 2:
            return 0
        bagWeight = (sum(nums) + target) // 2
        dp = [0] * (bagWeight + 1)      # 一维dp, 去掉物品i维度
        dp[0] = 1       # ?
        for i in range(len(nums)):
            for j in range(bagWeight, nums[i] - 1, -1):
                dp[j] += dp[j - nums[i]]
        return dp[-1]

    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        """ 474.一和零 """
        from collections import Counter

        # 时间:O(mxn) 空间:O(mxn)
        dp = [[0] * (n + 1) for _ in range(m + 1)]      # dp[i][j] 有i个0和j个1的最大子集长度为dp[i][j]
        for s in strs:
            zeroNum = Counter(s)['0']
            oneNum = Counter(s)['1']
            for i in range(m, zeroNum - 1, -1):
                for j in range(n, oneNum - 1, -1):
                    dp[i][j] = max(dp[i][j], dp[i - zeroNum][j - oneNum] + 1)
        return dp[-1][-1]
    
    def complete_bag(self):
        """ 完全背包理论基础 \n
            与01背包的差别:
                01背包递推公式  dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weights[i]] + values[i])
                完全背包递推公式 dp[i][j] = max(pd[i - 1][j], dp[i][j - weights[i]] + values[i])
            卡码网 52. 携带研究材料 """
        n, v = map(int, input().split())        # n物品数量 v背包容量
        weights = []
        values = []
        for _ in range(n):
            wi, vi = map(int, input().split())
            weights.append(wi)
            values.append(vi)
        dp = [[0] * (v + 1) for _ in range(n)]
        # for i in range(n):                      # 初始化首列, 虽然多余, 强调一下吧. 这句带着竟然会超时
        #     dp[i][0] = 0
        for j in range(weights[0], v + 1):      # 初始化首行
            dp[0][j] = dp[0][j - weights[0]] + values[0]
        for i in range(1, n):
            for j in range(v + 1):
                if j < weights[i]:
                    dp[i][j] = dp[i - 1][j]
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - weights[i]] + values[i])
        
        print(dp[-1][-1])

    def change(self, amount: int, coins: List[int]) -> int:
        """ 518.零钱兑换II \n
            01背包: 装满背包所得最大价值是多少      递推公式: dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weights[i]] + values[i]) \n
            纯完全背包: 装满背包所得最大价值是多少   递推公式: dp[i][j] = max(dp[i - 1][j], dp[i][j - weights[i]] + values[i]) \n
            01背包 组合:装满背包最大组合数         递推公式: dp[i][j] = dp[i - 1][j] + dp[i - 1][j - weights[i]] \n
            本题: 装满背包的组合数 (组合, 没有顺序) 递推公式: dp[i][j] = dp[i - 1][j] + dp[i][j - weights[i]] \n """
        # 时间:O(mxn) 空间:O(mxn)
        # dp = [[0] * (amount + 1) for _ in range(len(coins))]        # dp[i][j] 从[0, i]选物品, 装满容量为j的背包的组合数
        # for i in range(len(coins)):     # 初始化首列
        #     dp[i][0] = 1
        # for j in range(1, amount + 1):  # 初始化首行
        #     if j % coins[0] == 0:
        #         dp[0][j] = 1
        # for i in range(1, len(coins)):
        #     for j in range(amount + 1):
        #         if j < coins[i]:
        #             dp[i][j] = dp[i - 1][j]
        #         else:
        #             dp[i][j] = dp[i - 1][j] + dp[i][j - coins[i]]
        # return dp[-1][-1]

        ## 一维dp
        # 时间:O(mxn) 空间:O(n)
        dp = [0] * (amount + 1)     # dp[j] 装满容量为j的背包的组合数量(注: 组合没有顺序)
        dp[0] = 1
        for i in range(len(coins)):
            for j in range(coins[i], amount + 1):
                dp[j] += dp[j - coins[i]]       # 求 组合数/装满背包有多少种方法, 都是这个公式
        return dp[-1]
    
    def combinationSum4(self, nums: List[int], target: int) -> int:
        """ 377.组合总和IV \n
            完全背包 \n
            排列, 有顺序 """
        # 时间:O(mxn) 空间:O(n)
        dp = [0] * (target + 1)     # dp[j] 装满容量为j的背包有多少种方法 区分顺序 可重复拿
        dp[0] = 1
        for j in range(1, target + 1):
            for i in range(len(nums)):
                if nums[i] <= j:
                    dp[j] += dp[j - nums[i]]
        return dp[-1]
    
    def climb_(self):
        """ 卡码网 57.爬楼梯 进阶版 \n
            完全背包 \n
            排列, 有序 """
        # 提交正确
        n, m = map(int, input().split())        # n台阶数 m至多爬m阶
        dp = [0] * (n + 1)          # dp[j] 装满容量为j的背包有多少种方法 完全背包 有顺序
        dp[0] = 1
        for j in range(1, n + 1):
            for i in range(1, m + 1):   # 遍历物品 注意, 物品取值为1~m
                if i <= j:
                    dp[j] += dp[j - i]
        print(dp[-1])

    def coinChange(self, coins: List[int], amount: int) -> int:
        """ 322.零钱兑换 \n
            有点难度 好多细节 \n
            完全背包 \n
            求组合, 没顺序 --> 本题比较别致, 不关心组合还是排列, 要求的是所需最少硬币数量, 所以内外循环是谁都可以 """
        # 时间:O(mxn) 空间:O(n)
        # dp = [float('inf')] * (amount + 1)     # dp[j] 装满容量为j的背包的所需最少硬币数量 -- 同时思考递推公式
        # dp[0] = 0
        # for i in range(len(coins)):
        #     for j in range(coins[i], amount + 1):
        #         dp[j] = min(dp[j], dp[j - coins[i]] + 1)
        # return dp[-1] if dp[-1] != float('inf') else -1

        ## Again
        # 先遍历物品
        # dp = [float('inf')] * (amount + 1)    # dp[j] 装满容量为j的背包所需最少硬币数量
        # dp[0] = 0       # 递推基准
        # for i in range(len(coins)):
        #     for j in range(coins[i], amount + 1):
        #         dp[j] = min(dp[j], dp[j - coins[i]] + 1)
        # return dp[-1] if dp[-1] != float('inf') else -1

        # 先遍历背包
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        for j in range(1, amount + 1):
            for i in range(len(coins)):
                if coins[i] <= j:
                    dp[j] = min(dp[j], dp[j - coins[i]] + 1)
        return dp[-1] if dp[-1] != float('inf') else -1

    def numSquares(self, n: int) -> int:
        """ 279.完全平方数 \n
            由题意:
                完全背包 \n
                既不是组合也不是排列, 是求最小的数量, 所以内外遍历顺序无所谓 """
        # 时间:O(mxn) 空间:O(n)
        # dp = [float('inf')] * (n + 1)      # dp[j] 装满容量为j的背包所需完全平方数的最少数量为dp[j]
        # dp[0] = 0                          # 纯为递推 根据题意
        # for i in range(1, n + 1):
        #     for j in range(i ** 2, n + 1):
        #         dp[j] = min(dp[j], dp[j - i ** 2] + 1)
        # return dp[-1]

        ## 先遍历背包
        dp = [float('inf')] * (n + 1)
        dp[0] = 0       # 不容易想到
        for j in range(1, n + 1):
            for i in range(1, n + 1):
                if i ** 2 > j:
                    break
                dp[j] = min(dp[j], dp[j - i ** 2] + 1)
        return dp[-1]

    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        """ 139.单词拆分 \n
            完全背包 \n
            >>> 居然是求排列 <<< """
        # 时间:O(mxn) 空间:O(n)
        dp = [False] * (len(s) + 1)     # dp[j] 容量为j的背包能被单词字典装满
        dp[0] = True
        for j in range(1, len(s) + 1):
            for word in wordDict:
                length_word = len(word)
                if length_word <= j and dp[j - length_word] and s[j - length_word:j] in wordDict:
                    dp[j] = True
        return dp[-1]
    
    def multipleBags(self):
        """ 卡码网 56. 携带矿石资源 \n
            多重背包 转为 01背包 """
        # 提交超时, 不过你是懂其中原理的哈哈, 花了不少时间
        c, n = map(int, input().strip().split())            # c背包容量 n物品种类数量
        weights = list(map(int, input().strip().split()))
        values = [int(x) for x in input().strip().split() if x]
        nums = list(map(int, input().strip().split()))
        # _ws = []
        # _vs = []
        # for i, _n in enumerate(nums):
        #     _ws.extend([weights[i]] * _n)
        #     _vs.extend([values[i]] * _n)
        # weights = _ws
        # values = _vs
        dp = [0] * (c + 1)      # dp[j] 容量为j的背包最多能装下物品的价值为dp[j]
        for i in range(n):
            for j in range(c, weights[i] - 1, -1):
                for k in range(1, nums[i] + 1):         # 物品i最多能放nums[i], 这是在模拟能放多少个 --> 总的意思就是01背包, 你细品
                    if k * weights[i] > j:
                        break
                    dp[j] = max(dp[j], dp[j - k * weights[i]] + k * values[i])
        print(dp[-1])

    def rob(self, nums: List[int]) -> int:
        """ 198.打家劫舍 """
        n = len(nums)
        if n <= 2:
            return max(nums)
        dp = [0] * n            # dp[i] 偷0~i个屋子能得到的最大金钱数量为dp[i]
        dp[0] = nums[0]
        dp[1] = max(nums[:2])
        for i in range(2, n):
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
        return dp[-1]           # 回想dp[i]的意义, dp[i]已考虑所有偷窃方案, 所以就是最高的

    def rob(self, nums: List[int]) -> int:
        """ 213.打家劫舍II """
        # 时间:O(n) 空间:O(n)
        def rob_(nums, start, end):
            nums = nums[start:end + 1]
            n = len(nums)
            if n <= 2:
                return max(nums)
            dp = [0] * n        # dp[i] 打劫0~i房屋所得最多现金为dp[i]
            dp[0] = nums[0]
            dp[1] = max(nums[:2])
            for i in range(2, n):
                dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
            return dp[-1]
        
        if len(nums) <= 2:
            return max(nums)
        res1 = rob_(nums, 0, len(nums) - 2)
        res2 = rob_(nums, 1, len(nums) - 1)
        return max(res1, res2)
    
    def rob(self, root: Optional[TreeNode]) -> int:
        """ 337.打家劫舍III """
        # 时间:O(n) 所有节点遍历一次, 每个节点计算复杂度O(1)--即递归函数中 中节点 的计算
        # 空间:递归调用栈 平均情况 O(logn) 最坏时为单链结构 O(n)
        def backtrack(root):
            # 终止条件
            if root is None:
                return [0, 0]
            # 左
            left = backtrack(root.left)
            # 右
            right = backtrack(root.right)
            # 中
            ## 不偷当前节点
            val0 = max(left) + max(right)
            ## 偷当前节点
            val1 = root.val + left[0] + right[0]
            return [val0, val1]
        
        return max(backtrack(root))
    
    def maxProfit(self, prices: List[int]) -> int:
        """ 121.买卖股票的最佳时机 \n
            一次买卖 """
        # 时间:O(n) 空间:O(n)
        # n = len(prices)
        # dp = [[0] * 2 for _ in range(n)]        # dp[i][j]  第i天不持有/持有股票所得最大利润为dp[i][j]
        # dp[0] = [0, -prices[0]]
        # for i in range(1, n):
        #     dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
        #     dp[i][1] = max(dp[i - 1][1], -prices[i])
        # return dp[-1][0]

        ## 优化空间, 需要一定的优雅 -- 还是别作, 推荐上面那种
        n = len(prices)
        dp = [[0] * 2 for _ in range(2)]
        dp[0] = [0, -prices[0]]
        for i in range(1, n):
            dp[i % 2][0] = max(dp[(i - 1) % 2][0], dp[(i - 1) % 2][1] + prices[i])
            dp[i % 2][1] = max(dp[(i - 1) % 2][1], -prices[i])
        return dp[(n - 1) % 2][0]

    def maxProfit(self, prices: List[int]) -> int:
        """ 122.买卖股票的最佳时机II \n
            多次买卖 """
        # 时间:O(n) 空间:O(n)
        # n = len(prices)
        # dp = [[0] * 2 for _ in range(n)]        # dp[i][j] 第i天有两种状态 不持有/持有 的最大利润
        # dp[0] = [0, -prices[0]]
        # for i in range(1, n):
        #     dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
        #     dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])
        # return dp[-1][0]

        ## 优化空间
        n = len(prices)
        dp = [[0] * 2 for _ in range(2)]
        dp[0] = [0, -prices[0]]
        for i in range(1, n):
            dp[i % 2][0] = max(dp[(i - 1) % 2][0], dp[(i - 1) % 2][1] + prices[i])
            dp[i % 2][1] = max(dp[(i - 1) % 2][1], dp[(i - 1) % 2][0] - prices[i])
        return dp[(n - 1) % 2][0]
    
    def maxProfit(self, prices: List[int]) -> int:
        """ 123.买卖股票的最佳时机III """
        n = len(prices)
        dp = [[0] * 5 for _ in range(n)]
        for i in range(5):
            if i % 2:
                dp[0][i] = -prices[0]
        for i in range(1, n):
            for j in range(1, 5):
                if j % 2 == 0:  # 不持有
                    dp[i][j] = max(dp[i - 1][j - 1] + prices[i], dp[i - 1][j])
                else:           # 持有
                    dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - 1] - prices[i])
        return max(dp[-1])
    
    def maxProfit(self, k: int, prices: List[int]) -> int:
        """ 188.买卖股票的最佳时机IV """
        # n = len(prices)
        # dp = [[0] * (2 * k + 1) for _ in range(n)]
        # for i in range(1, 2 * k + 1):       # 状态0: 不操作, 也不用管
        #     if i % 2 == 1:      # 持有
        #         dp[0][i] = -prices[0]
        # for i in range(1, n):
        #     for j in range(1, 2 * k + 1):
        #         if j % 2 == 0:  # 不持有
        #             dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - 1] + prices[i])
        #         else:
        #             dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - 1] - prices[i])
        # return dp[-1][-1]

        ## 优化空间, 尝试一下, 也不难
        n = len(prices)
        dp = [0] * (2 * k + 1)
        for i in range(1, 2 * k + 1, 2):
            dp[i] = -prices[0]
        for i in range(1, n):
            for j in range(1, 2 * k + 1):
                if j % 2 == 0:      # 不持有
                    dp[j] = max(dp[j], dp[j - 1] + prices[i])
                else:               # 持有
                    dp[j] = max(dp[j], dp[j - 1] - prices[i])
        return dp[-1]
    
    def maxProfit(self, prices: List[int]) -> int:
        """ 309.买卖股票的最佳时机含冷冻期 """
        # 四个状态: 
        #   0 持有 
        #   1 不持有, 保持不持有 
        #   2 不持有, 今日卖出. 因为'冷冻期'前面只能是'今日卖出', 而不能是模糊的'不持有'
        #   3 冷冻期
        n = len(prices)
        dp = [[0] * 4 for _ in range(n)]
        dp[0] = [-prices[0], 0, 0, 0]
        for i in range(1, n):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] - prices[i], dp[i - 1][3] - prices[i])
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][3])
            dp[i][2] = dp[i - 1][0] + prices[i]
            dp[i][3] = dp[i - 1][2]
        return max(dp[-1])
    
    def maxProfit(self, prices: List[int], fee: int) -> int:
        """ 714.买买股票的最佳时机含手续费 """
        n = len(prices)
        dp = [[0] * 2 for _ in range(n)]
        dp[0] = [0, -prices[0]]
        for i in range(1, n):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i] - fee)
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])
        return dp[-1][0]

    def lengthOfLIS(self, nums: List[int]) -> int:
        """ 300.最长递增子序列 """
        # 时间:O(n^2) 空间:O(n)
        n = len(nums)
        dp = [1] * n        # dp[i] 以nums[i]结尾的最长递增子序列长度为dp[i]
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)
    
    def findLengthOfLCIS(self, nums: List[int]) -> int:
        """ 674.最长连续递增子序列 """
        n = len(nums)
        dp = [1] * n
        for i in range(1, n):
            if nums[i] > nums[i - 1]:
                dp[i] = dp[i - 1] + 1
        return max(dp)
    
    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        """ 718.最长重复子数组 \n
            看题意, 就是让找连续子序列 """
        # n1 = len(nums1)
        # n2 = len(nums2)
        # dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]    # dp[i][j] 以nums1[i - 1]结尾的子序列 和 以nums2[j - 1]结尾的子序列 的最长重复子数组的长度
        # for i in range(1, n1 + 1):
        #     for j in range(1, n2 + 1):
        #         if nums1[i - 1] == nums2[j - 1]:
        #             dp[i][j] = dp[i - 1][j - 1] + 1
        # return max(max(row) for row in dp)

        ## 优化空间 滚动数组
        m, n = len(nums1), len(nums2)
        dp = [0] * (n + 1)      # 意义不变 只是去掉了一个维度
        max_len = 0             # 最大值不一定出现在最后一行 所以要及时更新
        for i in range(1, m + 1):
            for j in range(n, 0, -1):
                if nums1[i - 1] == nums2[j - 1]:
                    dp[j] = dp[j - 1] + 1
                    max_len = max(max_len, dp[j])
                else:
                    dp[j] = 0
        return max_len
    
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        """ 1143.最长公共子序列 \
            不连续 """
        m = len(text1)
        n = len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return max(max(row) for row in dp)
    
    def maxUncrossedLines(self, nums1: List[int], nums2: List[int]) -> int:
        """ 1035.不相交的线 \n
            不连续, 同 1143.最长公共子序列 """
        m = len(nums1)
        n = len(nums2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]      # dp[i][j] 以nums1[i - 1]结尾的子序列 和 以nums2[j - 1]结尾的子序列 的最长公共子序列
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if nums1[i - 1] == nums2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return max(max(r) for r in dp)
    
    def maxSubArray(self, nums: List[int]) -> int:
        """ 53.最大子数组和 \n
            连续 """
        n = len(nums)
        dp = [0] * n        # dp[i] 以nums[i]为结尾的(连续)子数组 最大子数组和为dp[i]
        dp[0] = nums[0]     # 看dp[i]定义 dp[i]必须包括nums[i]
        for i in range(1, n):
            dp[i] = max(nums[i], dp[i - 1] + nums[i])
        return max(dp)
    
    def isSubsequence(self, s: str, t: str) -> bool:
        """ 392.判断子序列 \n
            不连续 同 1143.最长公共子序列 """
        # m = len(s)
        # n = len(t)
        # dp = [[0] * (n + 1) for _ in range(m + 1)]      # dp[i][j] 以s[i - 1]结尾的子序列 和 以t[j - 1]为结尾的子序列 的最长公共子序列长度为dp[i][j]
        # for i in range(1, m + 1):
        #     for j in range(1, n + 1):
        #         if s[i - 1] == t[j - 1]:
        #             dp[i][j] = dp[i - 1][j - 1] + 1
        #         else:
        #             dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        # return max(max(r) for r in dp) == m

        ## 编辑距离
        m = len(s)
        n = len(t)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s[i - 1] == t[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = dp[i][j - 1]     # 因为t长, 删除的话只能删除t
        return max(max(r) for r in dp) == m
    
    def numDistinct(self, s: str, t: str) -> int:
        """ 115.不同的子序列 \n
            编辑距离 """
        m = len(s)
        n = len(t)
        dp = [[0] * (n + 1) for _ in range(m + 1)]      # 以s[i - 1]结尾的子序列 中出现 以t[j - 1]结尾的子序列 的次数为dp[i][j]
        for i in range(m + 1):  # 首列
            dp[i][0] = 1        # 以s[i - 1]结尾的子序列 中 出现空串的次数, 当然是1. 不用考虑t的子序列 出现 s的子序列 的次数, 具体看答案
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s[i - 1] == t[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j]
                else:
                    dp[i][j] = dp[i - 1][j]
        return dp[-1][-1]
        
    def minDistance(self, word1: str, word2: str) -> int:
        """ 583.两个字符串的删除操作 """
        # m = len(word1)
        # n = len(word2)
        # dp = [[0] * (n + 1) for _ in range(m + 1)]      # dp[i][j] 使 以word1[i - 1]结尾 和 以word2[j - 1]结尾 的字符串变相同所需最小操作数
        # for i in range(1, m + 1):
        #     dp[i][0] = i
        # for j in range(1, n + 1):
        #     dp[0][j] = j
        # for i in range(1, m + 1):
        #     for j in range(1, n + 1):
        #         if word1[i - 1] == word2[j - 1]:
        #             dp[i][j] = dp[i - 1][j - 1]
        #         else:
        #             dp[i][j] = min(dp[i - 1][j - 1] + 2, dp[i - 1][j] + 1, dp[i][j - 1] + 1)
        # return dp[-1][-1]

        ## 1143.最长公共子序列 不连续
        m = len(word1)
        n = len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]      # dp[i][j] 以word1[i - 1]结尾 和 以word2[j - 1]结尾 的最长公共子序列长度为dp[i][j]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])
        return m + n - 2 * max(max(r) for r in dp)
    
    def minDistance(self, word1: str, word2: str) -> int:
        """ 72.编辑距离 """
        m = len(word1)
        n = len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]      # dp[i][j] 以word1[i - 1]结尾的单词 变为 以word2[j - 1]结尾的单词, 所需最小操作数
        for i in range(1, m + 1):
            dp[i][0] = i
        for j in range(1, n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j - 1] + 1, dp[i - 1][j] + 1, dp[i][j - 1] + 1)    # 替换操作没想到
        return dp[-1][-1]
    
    def countSubstrings(self, s: str) -> int:
        """ 647.回文子串\n
            连续 """
        # count = 0
        # n = len(s)
        # dp = [[False] * n for _ in range(n)]        # dp[i][j] 闭区间[i,j]子字符串(连续)是否为 回文子串
        # for i in range(n, -1, -1):
        #     for j in range(i, n):
        #         if s[i] == s[j]:
        #             if abs(i - j) <= 1:
        #                 dp[i][j] = True
        #                 count += 1
        #             elif dp[i + 1][j - 1]:
        #                 dp[i][j] = True
        #                 count += 1
        # return count

        ## 双指针法
        def extend(start, end, total):
            count = 0
            while start >= 0 and end < total:
                if s[start] == s[end]:
                    count += 1
                else:
                    break
                start -= 1
                end += 1
            return count
        
        n = len(s)
        count = 0
        for i in range(n):
            res1 = extend(i, i, n)
            res2 = extend(i, i + 1, n)
            count += (res1 + res2)
        return count
    
    def longestPalindromeSubseq(self, s: str) -> int:
        """ 516.最长回文子序列 \n
            区别:
                647.回文子串 连续
                本题, 不连续 """
        n = len(s)
        dp = [[0] * n for _ in range(n)]        # 闭区间[i,j]组成的回文子序列 的最大长度
        for i in range(n):
            dp[i][i] = 1
        for i in range(n - 1, -1, -1):
            for j in range(i + 1, n):
                if s[i] == s[j]:
                    dp[i][j] = dp[i + 1][j - 1] + 2
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j- 1])
        return max(max(r) for r in dp)
        
    
if __name__ == '__main__':
    """
    动归五部曲: 
        1.确定dp数组及下标i含义
        2.确定递推公式
        3.dp数组初始化
        4.确定遍历顺序
        4.举例推导
    """
    sl = Solution()

    s = "catsandog"
    wordDict = ["cats", "dog", "sand", "and", "cat"]
    print(sl.multipleBags())