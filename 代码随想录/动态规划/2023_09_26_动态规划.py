from typing import List, Optional


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def fib(self, n: int) -> int:
        """ 509.斐波那契数
            简单 """
        # 时间：O(n) 空间：O(n)
        # if n <= 1:
        #     return n
        # dp = [0] * (n + 1)
        # dp[0] = 0
        # dp[1] = 1
        # for i in range(2, n + 1):
        #     dp[i] = dp[i - 1] + dp[i - 2]
        # return dp[-1]

        # 优化空间：O(1)
        if n <= 1:
            return n
        dp = [0, 1]
        for i in range(2, n + 1):
            tmp = sum(dp)
            dp[0] = dp[1]
            dp[1] = tmp
        return dp[-1]

    def climbStairs(self, n: int) -> int:
        """ 70.爬楼梯
            简单 """
        # 时间：O(n) 空间：O(n)
        # if n == 1:
        #     return 1
        # dp = [0] * (n + 1)      # dp[i] 爬到索引第i阶有dp[i]种方法; 不使用dp[0] 因为题目已规定n范围
        # dp[1] = 1
        # dp[2] = 2
        # for i in range(3, n + 1):
        #     dp[i] = dp[i - 1] + dp[i - 2]
        # return dp[-1]

        # 优化空间复杂度：O(1)
        if n == 1:
            return 1
        dp = [1, 2]
        for i in range(2, n):   # 注意for范围和上面不同，灵活运用吧
            tmp = sum(dp)
            dp[0] = dp[1]
            dp[1] = tmp
        return dp[-1]

    def minCostClimbingStairs(self, cost: List[int]) -> int:
        """ 746.使用最小花费爬楼梯
            简单 """
        # 时间：O(n) 空间：O(n)
        # n = len(cost)
        # dp = [0] * (n + 1)      # dp[i] 爬到索引第i阶楼梯最小花费；dp[-1]即越过最后一阶台阶到达楼顶；注意，越过最后一个台阶，才叫到达楼顶(不要再想自己把'平地'当台阶0了)
        # for i in range(2, n + 1):
        #     dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])
        # return dp[-1]

        # 优化时间复杂度 O(1)
        n = len(cost)
        dp = [0, 0]     # 因为dp由前2个状态推导来
        for i in range(2, n + 1):
            tmp = min(dp[0] + cost[i - 2], dp[1] + cost[i - 1])
            dp[0] = dp[1]
            dp[1] = tmp
        return dp[-1]

        # 我这么想不太对，虽然能过，但dp意义不准确
        # n = len(cost)
        # dp = [0] * (n + 1)      # dp[i] 到达索引i台阶的最小花费 【注意】索引0台阶意味着还没有上台阶，还在地上呢
        # for i in range(2, n + 1):
        #     dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])
        # return dp[-1]

    def uniquePaths(self, m: int, n: int) -> int:
        """ 62.不同路径
            中等 """
        # 时间：O(m*n) 空间：O(m*n)
        # dp = [[1] * n] + [[1] + [0] * (n - 1) for _ in range(m - 1)]    # 首行、首列初始化为1
        # for i in range(1, m):
        #     for j in range(1, n):
        #         dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        # return dp[-1][-1]

        # 优化空间复杂度 O(n)
        dp = [1] * n
        for i in range(1, m):
            for j in range(1, n):
                dp[j] += dp[j - 1]
        return dp[-1]

    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        """ 63.不同路径II
            中等 """
        # 时间：O(m*n) 空间：O(m*n)
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        dp = [[0] * n for _ in range(m)]
        for i in range(m):
            if obstacleGrid[i][0] == 1:
                break
            dp[i][0] = 1
        for j in range(n):
            if obstacleGrid[0][j] == 1:
                break
            dp[0][j] = 1
        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] == 1:     # 遇到障碍 跳过
                    continue
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[-1][-1]

        # 优化空间复杂度，不信整不了
        # m, n = len(obstacleGrid), len(obstacleGrid[0])
        # dp = [0] * n        # 一维dp
        # for j in range(n):
        #     if obstacleGrid[0][j] == 1:
        #         break
        #     dp[j] = 1
        # for i in range(1, m):
        #     for j in range(n):
        #         if obstacleGrid[i][j] == 1:
        #             dp[j] = 0
        #         elif j != 0:
        #             dp[j] += dp[j - 1]
        # return dp[-1]

    def integerBreak(self, n: int) -> int:
        """ 343.整数拆分
            中等 """
        # 时间复杂度：O(n^2) 空间复杂度：O(n)
        dp = [0] * (n + 1)        # dp[i] 拆分i所得最大乘积
        dp[2] = 1       # dp[0] dp[1]不符合dp定义，不使用
        for i in range(3, n + 1):
            for j in range(1, i // 2 + 1):      # 从i拆分一个j来
                dp[i] = max(dp[i], j * (i - j), j * dp[i - j])
        return dp[-1]

    def numTrees(self, n: int) -> int:
        """ 96.不同的二叉搜索树
            中等 """
        dp = [0] * (n + 1)
        dp[0] = 1
        dp[1] = 1
        for i in range(2, n + 1):
            for j in range(i):
                dp[i] += dp[j] * dp[i - 1 - j]
        return dp[-1]

    def test_2_wei_bag_problem1(self):
        """ 01背包 导读1 """
        weights = [1, 3, 4]
        values = [15, 20, 30]
        bagweight = 4

        dp = [[0] * (bagweight + 1) for _ in range(len(weights))]
        for j in range(1, bagweight + 1):
            if j >= weights[0]:
                dp[0][j] = values[0]
        # 遍历
        for i in range(1, len(weights)):
            for j in range(1, bagweight + 1):
                if j >= weights[i]:
                    dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weights[i]] + values[i])
                else:
                    dp[i][j] = dp[i - 1][j]
        print(dp[-1][-1])

    def test_1_wei_bag_problem(self):
        """ 01背包 导读2 """
        weights = [1, 3, 4]
        values = [15, 20, 30]
        bagweight = 4

        dp = [0] * (bagweight + 1)
        for i in range(len(weights)):
            for j in range(bagweight, weights[i] - 1, -1):
                dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
        print(dp[-1])

    def canPartition(self, nums: List[int]) -> bool:
        """ 416.分割等和子集
            中等
            01背包：先遍历物品、再遍历容量且倒序 """
        # 时间：O(n^2) 空间：O(n)
        if sum(nums) % 2:       # 注意 % / 的区别，竟然错在这
            return False
        bagweight = sum(nums) // 2
        n = len(nums)
        dp = [0] * (bagweight + 1)
        for i in range(n):
            for j in range(bagweight, nums[i] - 1, -1):
                dp[j] = max(dp[j], dp[j - nums[i]] + nums[i])
        return dp[-1] == bagweight

    def lastStoneWeightII(self, stones: List[int]) -> int:
        """ 1049.最后一块石头的重量II
            中等
            01背包 先遍历物品、再遍历容量，内层for倒序
            dp[sum/2]能装下多少石头; 就是想求容量为sum(stones)//2的背包能放下重量为多少的石头 """
        # 时间：O(mxn) 空间：O(m)
        target = sum(stones) // 2
        dp = [0] * (target + 1)
        for i in range(len(stones)):
            for j in range(target, stones[i] - 1, -1):
                dp[j] = max(dp[j], dp[j - stones[i]] + stones[i])
        return (sum(stones) - dp[-1]) - dp[-1]

    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        """ 494.目标和
            中等
            要能推导出 Left = (target+sum)/2，就能想到背包
            组合问题--dp[j]+=dp[j-nums[i]] """
        # 时间：O(mxn) 空间：O(n)
        if (target + sum(nums)) % 2:
            return 0
        left = (target + sum(nums)) // 2
        dp = [0] * (left + 1)       # dp[j] 装满容量j的背包有几种方法; 排列问题
        dp[0] = 1
        for i in range(len(nums)):
            for j in range(left, nums[i] - 1, -1):
                dp[j] += dp[j - nums[i]]
        return dp[-1]

    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        """ 474.一和零
            中等 """
        # 时间：O(mxn) 空间：O(mxn)
        from collections import Counter

        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for s in strs:
            zeroNum = Counter(s)['0']
            oneNum = Counter(s)['1']
            for i in range(m, -1, -1):
                for j in range(n, -1, -1):
                    if zeroNum <= i and oneNum <= j:
                        dp[i][j] = max(dp[i][j], dp[i - zeroNum][j - oneNum] + 1)
        return dp[-1][-1]

    def test_CompletePack(self):
        """ 完全背包 导读 """
        weights = [1, 3, 4]
        values = [15, 20, 30]
        bagweight = 4

        dp = [0] * (bagweight + 1)
        for i in range(len(weights)):
            for j in range(weights[i], bagweight + 1):
                dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
        print(dp[-1])

    def change(self, amount: int, coins: List[int]) -> int:
        """ 518.零钱兑换II
            中等
            硬币数无限--完全背包
                        1.遍历顺序均可，2.内层遍历正序
            求组合数--dp[j] += dp[j-weight[i]] """
        # 时间：O(nxm) 空间：O(n)
        n = len(coins)
        dp = [0] * (amount + 1)
        dp[0] = 1
        for i in range(n):
            for j in range(coins[i], amount + 1):       # 一定注意范围，好几次错在这里
                dp[j] += dp[j - coins[i]]
        return dp[-1]

    def combinationSum4(self, nums: List[int], target: int) -> int:
        """ 377.组合总和IV
            中等
            元素个数无限--完全背包
            求组合个数 且 顺序不同视为不同结果--即排列，外层for容量、内层for物品且正序
            求排列、组合递推公式--dp[j] += dp[j - nums[i]] """
        # 时间：O(target * n) 空间：O(target)
        n = len(nums)
        dp = [0] * (target + 1)
        dp[0] = 1
        for j in range(1, target + 1):
            for i in range(n):
                if j >= nums[i]:
                    dp[j] += dp[j - nums[i]]
        return dp[-1]

    def climbStairsPro(self, n: int) -> int:
        """ 70.爬楼梯(进阶)
            简单
            之前做过 现在使用背包做
            元素不限--完全背包
            顺序不同的结果视为不同--求排列 外层for容量、内层for物品 """
        # 时间：O(nxm) 空间：O(n)
        if n == 1:
            return 1
        dp = [0] * (n + 1)
        dp[1] = 1
        dp[2] = 2
        weights = [1, 2]
        for i in range(3, n + 1):
            for j in range(len(weights)):
                dp[i] += dp[i - weights[j]]
        return dp[-1]

    def coinChange(self, coins: List[int], amount: int) -> int:
        """ 322.零钱兑换
            中等
            元素无限--完全背包
            不求 排列 or 组合，所以 外层for物品、内层for容量且正序
            求最小硬币个数，递推公式应为dp[j] = min(dp[j], dp[j-weight[i] + 1]) """
        # 时间：O(nxm) 空间：O(m)
        # n = len(coins)
        # dp = [float('inf')] * (amount + 1)
        # dp[0] = 0       # 注意 dp 意义
        # for i in range(n):
        #     for j in range(1, amount + 1):
        #         if j >= coins[i]:
        #             dp[j] = min(dp[j], dp[j - coins[i]] + 1)
        # return dp[-1] if dp[-1] != float('inf') else -1

        # 再写一遍
        """ 装满背包所需最少物品数量 递推公式dp[j]=min(dp[j], dp[j-weights[i]] + 1)
            物品无限--完全背包：外层、内层遍历顺序均可，内层遍历正序 """
        n = len(coins)
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        for i in range(n):
            for j in range(coins[i], amount + 1):
                dp[j] = min(dp[j], dp[j - coins[i]] + 1)
        return dp[-1] if dp[-1] != float('inf') else -1

    def numSquares(self, n: int) -> int:
        """ 279.完全平方数
            装满容量为j的背包所需最少物品数量 所以递推公式 dp[j] = min(dp[j], dp[j - i * i] + 1)
            物品数量不限--完全背包 1.遍历顺序均可 2.内层正序 """
        # 时间：O(n^3/2) 空间：O(n)
        if n == 1:
            return 1
        dp = [float('inf')] * (n + 1)
        dp[0] = 0       # 完全为了递推公式
        for i in range(1, n // 2 + 1):
            for j in range(i * i, n + 1):
                dp[j] = min(dp[j], dp[j - i * i] + 1)       # dp[j - i * i]会用到dp[0]，你想想
        return dp[-1]

    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        """ 139.单词拆分
            中等 """
        # 自己做的！
        # n = len(s)
        # dp = [False] * (n + 1)
        # dp[0] = True
        # for i in range(1, n + 1):
        #     for j in range(i, n + 1):
        #         if dp[i - 1] and s[i - 1:j] in wordDict:
        #             dp[j] = True
        # return dp[-1]

        # 完全背包
        """ 单词可重复用--完全背包，1.内、外层遍历顺序均可 2.内层遍历正序
            求排列，有顺序要求--外层背包、内层物品且正序 """
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True
        for i in range(1, n + 1):
            for j in range(i):
                if dp[j] and s[j:i] in wordDict:
                    dp[i] = True

    def test_multi_pack(self):
        """ 多重背包->01背包
            01背包 外层for物品、内层for容量且倒序 """
        # 示例答案也是90
        weights = [1, 3, 4]
        values = [15, 20, 30]
        nums = [2, 3, 2]
        bagweight = 10

        tmp_w =[]
        tmp_v = []
        for i in range(len(nums)):
            tmp_w += [weights[i]] * nums[i]
            tmp_v += [values[i]] * nums[i]
        weights = tmp_w
        values = tmp_v

        dp = [0] * (bagweight + 1)
        for i in range(len(weights)):
            for j in range(bagweight, weights[i] - 1, -1):
                dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
        print(dp[-1])

    def rob(self, nums: List[int]) -> int:
        """ 198.打家劫舍
            中等 """
        # 时间：O(n) 空间：O(n)
        # n = len(nums)
        # if n == 1:
        #     return nums[0]
        # dp = [0] * (n + 1)
        # dp[1] = nums[0]
        # dp[2] = max(nums[:2])
        # for i in range(3, n + 1):
        #     dp[i] = max(dp[i - 1], dp[i - 2] + nums[i - 1])
        # return dp[-1]

        # dp[i] 截止到索引i家，能抢劫到最大金额为dp[i]
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
        """ 213.打家劫舍II """
        def rob_(startInd, endInd):
            if startInd == endInd:
                return nums[startInd]
            dp = [0] * len(nums)
            dp[startInd] = nums[startInd]
            dp[startInd + 1] = max(nums[startInd:startInd + 2])
            for i in range(startInd + 2, endInd + 1):
                dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
            return dp[endInd]

        n = len(nums)
        if n == 1:
            return nums[0]
        rob1 = rob_(0, n - 2)
        rob2 = rob_(1, n - 1)
        return max(rob1, rob2)

    def robIII(self, root: Optional[TreeNode]) -> int:
        """ 337.打家劫舍III """
        # 时间：O(n) 空间：O(logn)
        def rob_(node):
            if not node:
                return [0, 0]       # 不偷/偷 当前节点
            lefts = rob_(node.left)
            rights = rob_(node.right)
            res1 = max(lefts) + max(rights)
            res2 = node.val + lefts[0] + rights[0]
            return (res1, res2)

        return max(rob_(root)[0], rob_(root)[1])

    def maxProfit(self, prices: List[int]) -> int:
        """ 121.买卖股票的最佳时机
            简单
            尝试动态规划 """
        # 动规五部曲
        # 时间：O(n)   空间：O(n)
        n = len(prices)
        dp = [[0, 0] for _ in range(n)]
        dp[0] = [0, -prices[0]]     # 不持有/持有
        for i in range(1, n):
            dp[i][0] = max(dp[i - 1][0], prices[i] + dp[i - 1][1])
            dp[i][1] = max(dp[i - 1][1], -prices[i])
        return max(dp[-1])      # 其实就是返回最后一天状态金额最大的--其实就是返回最后一天 不持有 状态时的所得金额

        # 先用最快的方法做出来
        # res = 0
        # minPrice = float('inf')
        # for i in range(len(prices)):
        #     minPrice = min(minPrice, prices[i])
        #     res = max(res, prices[i] - minPrice)
        # return res

    def maxProfitII(self, prices: List[int]) -> int:
        """ 122.买卖股票的最佳时机II
            中等
            无限次交易 """
        # 时间：O(n)   空间：O(n)
        n = len(prices)
        dp = [[0, 0] for _ in range(n)]     # 不持有/持有
        dp[0] = [0, -prices[0]]
        for i in range(1, n):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])
        return max(dp[-1])

    def maxProfitIII(self, prices: List[int]) -> int:
        """ 123.买卖股票的最佳时机III
            困难
            最多两笔交易 """
        # 时间：O(n)   空间：O(5n)
        n = len(prices)
        dp = [[0] * 5 for _ in range(n)]
        dp[0] = [0, -prices[0], 0, -prices[0], 0]
        for i in range(1, n):
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])
            dp[i][2] = max(dp[i - 1][2], dp[i - 1][1] + prices[i])
            dp[i][3] = max(dp[i - 1][3], dp[i - 1][2] - prices[i])
            dp[i][4] = max(dp[i - 1][4], dp[i - 1][3] + prices[i])
        return max(dp[-1])

    def maxProfitIV(self, k: int, prices: List[int]) -> int:
        """ 188.买卖股票的最佳时机IV
            困难
            买卖k次 """
        # 时间：O(k*n)   空间：O(k*n)
        n = len(prices)
        dp = [[0] * (2 * k + 1) for _ in range(n)]
        for j in range(1, 2 * k + 1):
            if j % 2:
                dp[0][j] = -prices[0]
        for i in range(1, n):
            for j in range(1, 2 * k + 1):
                if j % 2:
                    dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - 1] - prices[i])
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - 1] + prices[i])
        return max(dp[-1])

    def maxProfitFreeze(self, prices: List[int]) -> int:
        """ 309.买卖股票的最佳时机含冷冻期
            中等
            状态：
                状态0.持有
                状态1.不持有，在冷冻期
                状态2.不持有，不在冷冻期 """
        n = len(prices)
        dp = [[0] * 3 for _ in range(n)]
        dp[0] = [-prices[0], 0, 0]
        for i in range(1, n):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][2] - prices[i])
            dp[i][1] = dp[i - 1][0] + prices[i]
            dp[i][2] = max(dp[i - 1][2], dp[i - 1][1])
        return max(dp[-1])

    def maxProfitFee(self, prices: List[int], fee: int) -> int:
        """ 714.买卖股票的最佳时机含手续费
            题目说明无限次交易 """
        # 时间：O(n)   空间：O(n)
        n = len(prices)
        dp = [[0] * 2 for _ in range(n)]
        dp[0] = [0, -prices[0]]     # 不持有/持有
        for i in range(1, n):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i] - fee)
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])
        return max(dp[-1])

    def lengthOfLIS(self, nums: List[int]) -> int:
        """ 300.最长递增子序列
            中等 """
        # 时间：O(n^2) 空间：O(n)
        n = len(nums)
        dp = [1] * n
        for i in range(n):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[j] + 1, dp[i])
        return max(dp)

    def findLengthOfLCIS(self, nums: List[int]) -> int:
        """ 674.最长连续递增子序列
            简单 """
        n = len(nums)
        dp = [1] * n
        for i in range(1, n):
            if nums[i - 1] < nums[i]:
                dp[i] = dp[i - 1] + 1
        return max(dp)

    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        """ 718.最长重复子数组
            中等
            其实是求 最长的连续子数组 长度 """
        # 时间：O(m x n)   空间：O(m x n)
        # m, n = len(nums1), len(nums2)
        # dp = [[0] * (n + 1) for _ in range(m + 1)]
        # for i in range(1, m + 1):
        #     for j in range(1, n + 1):
        #         if nums1[i - 1] == nums2[j - 1]:
        #             dp[i][j] = dp[i - 1][j - 1] + 1
        # return max(max(ls) for ls in dp)

        # 再写一遍
        m, n = len(nums1), len(nums2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if nums1[i - 1] == nums2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
        return max(max(ls) for ls in dp)

        # 尝试优化空间复杂度--失败
        # m, n = len(nums1), len(nums2)
        # dp = [0] * (n + 1)
        # for i in range(1, m + 1):
        #     for j in range(n, 0, -1):
        #         if nums1[i - 1] == nums2[j - 1]:
        #             dp[j] = dp[j - 1] + 1
        #         else:
        #             dp[j] = 0
        # return max(dp)

    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        """ 1143.最长公共子序列
            中等
            不要求 '连续' """
        # m, n = len(text1), len(text2)
        # dp = [[0] * (n + 1) for _ in range(m + 1)]
        # for i in range(1, m + 1):
        #     for j in range(1, n + 1):
        #         if text1[i - 1] == text2[j - 1]:
        #             dp[i][j] = dp[i - 1][j - 1] + 1
        #         else:
        #             dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])
        # return max(max(ls) for ls in dp)

        # 重写一遍
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])
        return dp[-1][-1]

    def maxUncrossedLines(self, nums1: List[int], nums2: List[int]) -> int:
        """ 1035.不相交的线
            中等
            即 1143.最长公共子序列，一模一样 """
        m, n = len(nums1), len(nums2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if nums1[i - 1] == nums2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])
        return max(max(ls) for ls in dp)

    def maxSubArray(self, nums: List[int]) -> int:
        """ 53.最大子数组和
            中等 """
        # 时间：O(n)   空间：O(n)
        n = len(nums)
        dp = [0] * n
        dp[0] = nums[0]
        for i in range(1, n):
            dp[i] = max(dp[i - 1] + nums[i], nums[i])
        return max(dp)

    def isSubsequence(self, s: str, t: str) -> bool:
        """ 392.判断子序列
            简单 不简单
            -> 不连续子序列：dp[i]==dp[j], dp[i][j] = dp[i-1][j-1]+1; dp[i]!=dp[j], dp[i][j]=max(dp[i][j-1], dp[i-1][j])
            ->   连续子序列：dp[i]==dp[j], dp[i][j] = dp[i-1][j-1]+1 """
        # m, n = len(s), len(t)
        # dp = [[0] * (n + 1) for _ in range(m + 1)]      # dp[i][j] 以索引i-1结尾的s、以j-1结尾的t 的 最长子序列长度
        # for i in range(1, m + 1):
        #     for j in range(1, n + 1):
        #         if s[i - 1] == t[j - 1]:
        #             dp[i][j] = dp[i - 1][j - 1] + 1
        #         else:
        #             dp[i][j] = dp[i][j - 1]
        # return dp[-1][-1] == m

        # 再写一遍
        # 判断s是否为t的子序列，删除t
        m, n = len(s), len(t)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s[i - 1] == t[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = dp[i][j - 1]
        return dp[-1][-1] == m

    def numDistinct(self, s: str, t: str) -> int:
        """ 115.不同的子序列
            困难 """
        # m, n = len(s), len(t)
        # dp = [[0] * (n + 1) for _ in range(m + 1)]      # dp[i][j] 以索引i-1元素结尾的s的子序列 中 出现索引j-1元素结尾的t 的次数
        # # dp初始化
        # for i in range(m + 1):
        #     dp[i][0] = 1        # 以索引i-1元素结尾的s子序列 中出现 空串的次数，只能1种
        # for j in range(n + 1):
        #     dp[0][j] = 0        # 空串 中出现 以索引j-1元素结尾的t 的次数，无法出现，所以是0种
        # dp[0][0] = 1            # 空串 中出现 空串 的次数，1种
        # # 遍历计算
        # for i in range(1, m + 1):
        #     for j in range(1, n + 1):
        #         # 子序列问题，基本就是2种情况 1.当前元素相等 2.当前元素不相等
        #         if s[i - 1] == t[j - 1]:
        #             dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j]
        #         else:
        #             dp[i][j] = dp[i - 1][j]
        # return dp[-1][-1]

        # 再写一遍
        m, n = len(s), len(t)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = 1
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s[i - 1] == t[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j]
                else:
                    dp[i][j] = dp[i - 1][j]
        return dp[-1][-1]

    def minDistance(self, word1: str, word2: str) -> int:
        """ 583.两个字符串的删除操作
            中等 """
        # m, n = len(word1), len(word2)
        # dp = [[0] * (n + 1) for _ in range(m + 1)]
        # # 初始化dp首行首列
        # for i in range(m + 1):
        #     dp[i][0] = i
        # for j in range(n + 1):
        #     dp[0][j] = j
        # for i in range(1, m + 1):
        #     for j in range(1, n + 1):
        #         if word1[i - 1] == word2[j - 1]:
        #             dp[i][j] = dp[i - 1][j - 1]
        #         else:
        #             dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1)
        # return dp[-1][-1]

        # 再写一遍
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            dp[i][0] = i
        for j in range(1, n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i][j - 1] + 1, dp[i - 1][j] + 1)
        return dp[-1][-1]

        # 同 1143.最长公共子序列
        # m, n = len(word1), len(word2)
        # dp = [[0] * (n + 1) for _ in range(m + 1)]
        # for i in range(1, m + 1):
        #     for j in range(1, n + 1):
        #         if word1[i - 1] == word2[j - 1]:
        #             dp[i][j] = dp[i - 1][j - 1] + 1
        #         else:
        #             dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        # return m + n - 2 * dp[-1][-1]

    def minDistance_super(self, word1: str, word2: str) -> int:
        """ 72.编辑距离
            困难 """
        # m, n = len(word1), len(word2)
        # dp = [[0] * (n + 1) for _ in range(m + 1)]
        # for i in range(m + 1):
        #     dp[i][0] = i
        # for j in range(n + 1):
        #     dp[0][j] = j
        # for i in range(1, m + 1):
        #     for j in range(1, n + 1):
        #         if word1[i - 1] == word2[j - 1]:
        #             dp[i][j] = dp[i - 1][j - 1]
        #         else:
        #             dp[i][j] = min(dp[i][j - 1] + 1, dp[i - 1][j] + 1,      # 增/删  增即删，删即增
        #                            dp[i - 1][j - 1] + 1)                    # 替换
        # return dp[-1][-1]

        # 再写一遍
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            dp[i][0] = i
        for j in range(1, n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1)
        return dp[-1][-1]

    def countSubstrings(self, s: str) -> int:
        """ 647.回文子串
            中等 """
        # n = len(s)
        # dp = [[False] * n for _ in range(n)]
        # for i in range(n):
        #     dp[i][i] = True
        # for i in range(n - 1, -1, -1):
        #     for j in range(i, n):
        #         if s[i] == s[j]:
        #             if j - i <= 1:
        #                 dp[i][j] = True
        #             elif dp[i + 1][j - 1]:
        #                 dp[i][j] = True
        # return sum(sum(ls) for ls in dp)

        # 再写一遍
        # n = len(s)
        # dp = [[False] * n for i in range(n)]
        # for i in range(n):
        #     dp[i][i] = True
        # for i in range(n - 1, -1, -1):
        #     for j in range(i + 1, n):
        #         if s[i] == s[j]:
        #             if j - i == 1:
        #                 dp[i][j] = True
        #             elif dp[i + 1][j - 1]:
        #                 dp[i][j] = True
        # return sum(sum(ls) for ls in dp)

        # 再写一遍
        # 包含'连续', 用True/False表示
        n = len(s)
        dp = [[False] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = True
        for i in range(n - 1, -1, -1):
            for j in range(i + 1, n):
                if s[i] == s[j]:
                    if j - i == 1:
                        dp[i][j] = True
                    elif dp[i + 1][j - 1]:
                        dp[i][j] = True
        return sum(sum(ls) for ls in dp)

    def longestPalindromeSubseq(self, s: str) -> int:
        """ 516.最长回文子序列
            中等 """
        # n = len(s)
        # dp = [[0] * n for _ in range(n)]
        # for i in range(n):
        #     dp[i][i] = 1
        # for i in range(n - 1, -1, -1):
        #     for j in range(i + 1, n):
        #         if s[i] == s[j]:
        #             dp[i][j] = dp[i + 1][j - 1] + 2
        #         else:
        #             dp[i][j] = max(dp[i][j - 1], dp[i + 1][j])
        # return max(max(ls) for ls in dp)

        # 再写一遍
        # 不包含'连续'，则dp[i][j]=具体长度
        n = len(s)
        dp = [[0] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = 1
        for i in range(n - 1, -1, -1):
            for j in range(i + 1, n):
                if s[i] == s[j]:
                    dp[i][j] = dp[i + 1][j - 1] + 2
                else:
                    dp[i][j] = max(dp[i][j - 1], dp[i + 1][j])
        return max(max(ls) for ls in dp)


if __name__ == "__main__":
    sl = Solution()

    s = "cbbd"
    print(sl.longestPalindromeSubseq(s))
