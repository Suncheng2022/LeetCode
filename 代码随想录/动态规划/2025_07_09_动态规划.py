"""
07.09   是的, 该走了
"""
from typing import List

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

    nums = [1, 1, 1, 1]
    target = -1000
    print(sl.findTargetSumWays(nums, target))