from typing import List, Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def test_2_wei_bag_problem1(self):
        """ 01背包导读1 二维数组实现 """
        weights = [1, 3, 4]
        values = [15, 20, 30]
        bagweight = 4

        # 定义dp
        dp = [[0] * (bagweight + 1) for _ in range(len(values))]
        # 初始化dp
        for i in range(len(values)):
            dp[i][0] = 0
        for j in range(bagweight + 1):
            dp[0][j] = values[0] if j >= weights[0] else 0
        # 遍历：二维dp遍历顺序均可；
        #      先遍历物品、再遍历容量比较好理解
        for i in range(1, len(weights)):        # 使用 索引0物品 已经用于初始化了
            for j in range(bagweight + 1):
                if j < weights[i]:              # 不放 weights[i]
                    dp[i][j] = dp[i - 1][j]
                else:                           # 放 weights[i]
                    dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weights[i]] + values[i])
        return dp[-1][-1]

    def test_1_wei_bag_problem(self):
        """ 01背包导读2 一维数组实现 """
        weights = [1, 3, 4]
        values = [15, 20, 30]
        bagweight = 4

        # 定义一维dp
        dp = [0] * (bagweight + 1)
        # 遍历：一维dp，必须先遍历物品、再遍历容量且倒序
        for i in range(len(weights)):
            for j in range(bagweight, weights[i] - 1, -1):       # dp[0] = 0, 这是包含在初始化中的
                dp[j] = max(dp[j], dp[j - weights[i]] + values[i])     # 一维dp去掉的是维度i
        return dp[-1]

    def minCostClimbingStairs(self, cost: List[int]) -> int:
        """ 746.使用最小花费爬楼梯 """
        dp = [0] * (len(cost) + 1)        # dp[i] 爬到索引i台阶所需花费
        for i in range(2, len(cost) + 1):
            dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])
        return dp[-1]

    def uniquePaths(self, m: int, n: int) -> int:
        """ 62.不同路径 """
        dp = [[1] * n] + [[1] + [0] * (n - 1) for _ in range(m - 1)]
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[-1][-1]

    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        """ 63.不同路径II """
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        dp = [[0] * n for _ in range(m)]
        # 初始化首行
        for j in range(n):
            if obstacleGrid[0][j] == 1:
                break
            dp[0][j] = 1
        # 初始化首列
        for i in range(m):
            if obstacleGrid[i][0] == 1:
                break
            dp[i][0] = 1
        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] == 1:
                    continue
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[-1][-1]

    def integerBreak(self, n: int) -> int:
        """ 343.整数拆分 """
        dp = [0] * (n + 1)      # dp[i] 表示将数字i拆分所得最大乘积
        # 初始化 dp[0] dp[1]根本无法拆分，所以不使用
        dp[2] = 1
        for i in range(3, n + 1):
            for j in range(1, i // 2 + 1):      # 从i上拆下来一个j，所以j不为0，而且尽可能拆分成几个相同的数
                dp[i] = max(dp[i], j * (i - j), j * dp[i - j])      # 递推公式不好想，记住吧
        return dp[-1]

    def numTrees(self, n: int) -> int:
        """ 96.不同的二叉搜索树 """
        dp = [0] * (n + 1)
        # 初始化dp[0]、dp[1]，空 也是一种二叉搜索树
        dp[0] = 1
        dp[1] = 1       # 题目要求 1 <= n
        for i in range(2, n + 1):       # 以i为根节点、计算dp[i]
            for j in range(i):          # 其中1棵子树的节点数为j，则 0<=j<=i-1
                dp[i] += dp[j] * dp[i - 1 - j]
        return dp[-1]

    def canPartition(self, nums: List[int]) -> bool:
        """ 416.分割等和子集
            分成2堆完全相等的
            01背包 一维dp--先遍历物品、再遍历容量且倒序；内层for倒序
            元素不能重复使用--01背包，否则完全背包 """
        if sum(nums) % 2:
            return False
        target = sum(nums) // 2
        dp = [0] * (target + 1)         # dp[i] 容量为i的背包所能得到的最大价值
        for i in range(len(nums)):
            for j in range(target, nums[i] - 1, -1):
                dp[j] = max(dp[j], dp[j - nums[i]] + nums[i])
        return dp[-1] == target

    def lastStoneWeightII(self, stones: List[int]) -> int:
        """ 1049.最后一块石头的重量II
            分成2堆大致相等的
            01背包，元素不能重复用
            一维dp，先遍历物品、再遍历容量且倒序/或说 一维dp则内层for倒序 """
        target = sum(stones) // 2
        dp = [0] * (target + 1)         # dp[i] 容量为i的背包能装下最大重量是多少
        for i in range(len(stones)):
            for j in range(target, stones[i] - 1, -1):
                dp[j] = max(dp[j], dp[j - stones[i]] + stones[i])
        return sum(stones) - 2 * dp[-1]

    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        """ 494.目标和【组合问题】
            推导:
                left - right = target
                left + right = sum
            得 left = (target + sum) / 2
            之前题目求，容量为j的背包，能装下的最大价值为dp[j]
            本题，装满容量为left的背包有dp[left]种方法，是求组合问题，其递归公式均为 dp[j] += dp[j-nums[i]] """
        if (target + sum(nums)) % 2 or abs(target) > sum(nums):
            return 0
        left = (target + sum(nums)) // 2    # 求装满容量为left的背包有几种方法
        dp = [0] * (left + 1)               # dp[j] 装满容量为j的背包 有dp[j]种方法
        dp[0] = 1
        for i in range(len(nums)):
            for j in range(left, nums[i] - 1, -1):
                dp[j] += dp[j - nums[i]]
        return dp[-1]

    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        """ 474.一和零
            依然是01背包，只不过物品有了2个维度的重量而已，value即字符串的个数——即每个字符串的价值都是 1。按照01背包公式来就行 """
        from collections import Counter

        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for s in strs:
            zeroNum = Counter(s)['0']
            oneNum = Counter(s)['1']
            for i in range(m, zeroNum - 1, -1):
                for j in range(n, oneNum - 1, -1):
                    dp[i][j] = max(dp[i][j], dp[i - zeroNum][j - oneNum] + 1)
        return dp[-1][-1]

    """ ----------------- 完全背包 -----------------
        与 01背包 最大的不同就是物品数量无限->同时，内层for正序即可 """
    def test_CompletePack(self):
        """ 完全背包 导读 """
        weights = [1, 3, 4]
        values = [15, 20, 30]
        bagWeight = 4

        dp = [0] * (bagWeight + 1)
        for i in range(len(weights)):
            for j in range(weights[i], bagWeight + 1):
                dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
        print(dp[-1])

    def change(self, amount: int, coins: List[int]) -> int:
        """ 518.零钱兑换II  【组合问题】 如，装满背包有几种方法
            首先确定 组合问题--递推公式 dp[j] += dp[j-nums[i]]
                            组合问题似乎要初始化dp[0]
                            组合问题，外层for物品、内层for容量[正序]
            硬币无限使用--完全背包:
                        物品、容量的遍历顺序均可，但本题不可以（虽然我写对了）！"""
        dp = [0] * (amount + 1)     # dp[j] 组成金额j的硬币组合数
        dp[0] = 1
        for i in range(len(coins)):
            for j in range(coins[i], amount + 1):
                dp[j] += dp[j - coins[i]]
        return dp[-1]

    def combinationSum4(self, nums: List[int], target: int) -> int:
        """ 377.组合总和IV 结果顺序不同视为不同解
            排列问题--由 组合问题 进化而来
                组合问题：
                    递推公式 dp[j] += dp[j - nums[i]]
                    初始化dp[0]=1
                    外层for物品、内层for容量[正序]
                排列问题：与 组合问题 仅内外层遍历顺序不同
                    递归公式 同上
                    初始化 同上
                    *外层for容量、内层for物品 """
        dp = [0] * (target + 1)         # dp[j] 总和为j的背包的元素组合的个数，即排列数
        dp[0] = 1
        # 排列问题，外层for容量、内层for物品
        for i in range(1, target + 1):
            for j in range(len(nums)):
                if i >= nums[j]:
                    dp[i] += dp[i - nums[j]]
        return dp[-1]

    def climbStairs_(self, n: int) -> int:
        """ 70.爬楼梯 使用背包解法
            n--背包容量
            每次爬1、2个台阶--物品
            物品可重复使用--完全背包
            爬到楼顶有多少种方法，暗含顺序--求排列，由 组合问题 进化而来
                                                组合问题：
                                                    1.初始化 dp[0] = 1
                                                    2.递推公式 dp[j] += dp[j - nums[i]]
                                                    3.外层for物品、内层for容量
                                                排列问题：
                                                    1.同上
                                                    2.同上
                                                    *3.外层for容量、内层for物品"""
        dp = [0] * (n + 1)      # dp[i] 爬到n阶有多少种方法
        dp[0] = 1               # 初始化
        for i in range(1, n + 1):       # 外层for容量
            for j in range(1, 2 + 1):   # 内层for物品
                if i >= j:
                    dp[i] += dp[i - j]
        return dp[-1]

    def coinChange(self, coins: List[int], amount: int) -> int:
        """ 322.零钱兑换 最小硬币个数
            硬币无限--完全背包
            不在乎顺序--内外层for遍历顺序均可，那就按组合问题来吧
            递推公式 dp[i] = min(dp[i], dp[i - nums[j]] + 1)，因为要求最小硬币数嘛 """
        # dp = [float('inf')] * (amount + 1)     # dp[i] 金额总和为i所需最小硬币数
        # dp[0] = 0                   # 初始化
        # for i in range(len(coins)):
        #     for j in range(coins[i], amount + 1):
        #         dp[j] = min(dp[j], dp[j - coins[i]] + 1)
        # return dp[-1] if dp[-1] != float('inf') else -1

        dp = [float('inf')] * (amount + 1)      # dp[i] 组成金额i所需最小硬币个数
        dp[0] = 0       # 初始化，组成金额0需要最少0个硬币
        for i in range(len(coins)):
            for j in range(coins[i], amount + 1):
                dp[j] = min(dp[j], dp[j - coins[i]] + 1)
        return dp[-1] if dp[-1] != float('inf') else -1

    def numSquares(self, n: int) -> int:
        """ 279.完全平方数 和为n所需最少的完全平方数 """
        # 重新写一遍 yes!
        dp = [float('inf')] * (n + 1)       # dp[i]  和为i所需最少的完全平方数个数
        dp[0] = 0                           # 仅为递推公式
        for i in range(1, int(n ** .5) + 1):    # 遍历物品
            for j in range(i ** 2, n + 1):      # 遍历容量
                dp[j] = min(dp[j], dp[j - i ** 2] + 1)
        return dp[-1] if dp[-1] != float('inf') else 0

        # dp = [float('inf')] * (n + 1)       # dp[i] 和为i所需最少完全平方数的个数
        # dp[0] = 0       # 由题意，完全平方数不包括0，仅为递推
        # for i in range(1, int(n ** .5) + 1):        # 遍历物品
        #     for j in range(i ** 2, n + 1):               # 遍历容量
        #         dp[j] = min(dp[j], dp[j - i ** 2] + 1)
        # return int(dp[-1]) if dp[-1] != float('inf') else 0

        # dp = [float('inf')] * (n + 1)
        # dp[0] = 0
        # dp[1] = 1
        # for i in range(1, n + 1):
        #     for j in range(i * i, n + 1):
        #         dp[j] = min(dp[j], dp[j - i * i] + 1)
        # return int(dp[-1])

    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        """ 139.单词拆分
            能确定是 完全背包
            本题其实是求排列--先遍历背包、再遍历物品 """
        dp = [False] * (len(s) + 1)     # dp[i] 前i个字符串能否由wordDict表示
        dp[0] = True                    # 纯为递推
        for i in range(1, len(s) + 1):
            for j in range(i):
                if dp[j] and s[j:i] in wordDict:
                    dp[i] = True
                    break
        return dp[-1]

    def test_multi_pack(self):
        """ 多重背包 导读
            转成01背包 """
        weights = [1, 3, 4]
        values = [15, 20, 30]
        nums = [2, 3, 2]
        bagWeight = 10

        weights = [n for ls in [[w] * n for w, n in zip(weights, nums)] for n in ls]
        values = [n for ls in [[v] * n for v, n in zip(values, nums)] for n in ls]

        dp = [0] * (bagWeight + 1)
        for i in range(len(weights)):
            for j in range(bagWeight, weights[i] - 1, -1):
                dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
        print(dp[-1])

    def rob(self, nums: List[int]) -> int:
        """ 198.打家劫舍 """
        # 《代码随想录》，但好像我自己写的更健壮
        n = len(nums)
        dp = [0] * n        # dp[i] 截止到当前索引i房间，所能偷到的最大金额
        dp[0] = nums[0]
        dp[1] = max(nums[:2])
        for i in range(2, n):
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
        return dp[-1]

        # 我这种定义dp，其索引似乎不好理解
        # n = len(nums)
        # dp = [0] * (n + 1)      # dp[i] 偷前i间房屋所得最大金额
        # dp[1] = nums[0]         # 初始化
        # for i in range(2, n + 1):
        #     dp[i] = max(dp[i - 1], dp[i - 2] + nums[i - 1])
        # return dp[-1]

    def robII(self, nums: List[int]) -> int:
        """ 213.打家劫舍II 成圈了 """
        def rob_(nums, startInd, endInd):
            """ 闭区间[startInd, endInd] """
            nums = nums[startInd:endInd + 1][:]
            n = len(nums)
            dp = [0] * (n + 1)      # dp[i] 前i间房屋能偷到的最大金额
            dp[1] = nums[0]
            for i in range(2, n + 1):
                dp[i] = max(dp[i - 1], dp[i - 2] + nums[i - 1])
            return dp[-1]
        if len(nums) == 1:
            return nums[0]
        res1 = rob_(nums, 0, len(nums) - 2)
        res2 = rob_(nums, 1, len(nums) - 1)
        return max(res1, res2)

    def robIII(self, root: Optional[TreeNode]) -> int:
        """ 337.打家劫舍III """
        def backtracking(node):
            if not node:
                return [0, 0]       # [不偷,偷]    所得均为0
            lefts = backtracking(node.left)         # 返回 [不偷,偷]
            rights = backtracking(node.right)       # 返回 [不偷,偷]
            return [max(lefts) + max(rights),
                    node.val + lefts[0] + rights[0]]

        res = backtracking(root)                    # 返回 [不偷,偷]
        return max(res)

    def maxProfit(self, prices: List[int]) -> int:
        """ 121.买卖股票的最佳时机
            只卖买一次 """
        n = len(prices)
        dp = [[0, 0] for _ in range(n)]     # 不持有/持有
        dp[0] = [0, -prices[0]]
        for i in range(1, n):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
            dp[i][1] = max(dp[i - 1][1], -prices[i])                    # 只能买卖一次，体现在递推公式——买的时候就是'第一次'买，因为仅一次买卖没有盈余呀
        return max(dp[-1])

    def maxProfitII(self, prices: List[int]) -> int:
        """ 122.买卖股票的最佳时机II
            可买卖多次 """
        n = len(prices)
        dp = [[0, 0] for _ in range(n)]     # 不持有/持有
        dp[0] = [0, -prices[0]]
        for i in range(1, n):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])      # 可多次买卖，体现在递推公式——买的时候有盈余了，因为可多次买卖前面可能有盈余
        return dp[-1][0]

    def maxProfitIII(self, prices: List[int]) -> int:
        """ 123.买卖股票的最佳时机III
            最多2次交易
                状态0：无操作
                状态1：第一次持有
                状态2：第一次不持有
                状态3：第二次持有
                状态4：第二次不持有 """
        n = len(prices)
        dp = [[0] * 5 for _ in range(n)]
        dp[0] = [0, -prices[0], 0, -prices[0], 0]
        for i in range(1, n):
            dp[i][1] = max(dp[i - 1][1], -prices[i])
            dp[i][2] = max(dp[i - 1][2], dp[i - 1][1] + prices[i])
            dp[i][3] = max(dp[i - 1][3], dp[i - 1][2] - prices[i])
            dp[i][4] = max(dp[i - 1][4], dp[i - 1][3] + prices[i])
        return max(dp[-1])

    def maxProfitIV(self, k: int, prices: List[int]) -> int:
        """ 188.买卖股票的最佳时机IV
            最多k次交易 """
        n = len(prices)
        dp = [[0] * (2 * k + 1) for _ in range(n)]
        for j in range(2 * k + 1):
            if j % 2:
                dp[0][j] = -prices[0]
        for i in range(1, n):
            for j in range(1, 2 * k + 1):
                if j % 2:       # 奇数，持有
                    dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - 1] - prices[i])
                else:           # 偶数，不持有
                    dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - 1] + prices[i])
        return max(dp[-1])

    def maxProfit_Freeze(self, prices: List[int]) -> int:
        """ 309.买卖股票的最佳时机含冷冻期
            状态0：持有
            状态1：不持有 在冷冻期
            状态2：不持有 不在冷冻期 """
        n = len(prices)
        dp = [[0] * 3 for _ in range(n)]
        dp[0] = [-prices[0], 0, 0]
        for i in range(1, n):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][2] - prices[i])
            dp[i][1] = dp[i - 1][0] + prices[i]
            dp[i][2] = max(dp[i - 1][2], dp[i - 1][1])
        return max(dp[-1])

    def maxProfit_Fee(self, prices: List[int], fee: int) -> int:
        """ 714.买买股票的最佳时机含手续费
            无限次交易，有手续费而已 """
        n = len(prices)
        dp = [[0] * 2 for _ in range(n)]
        dp[0] = [0, -prices[0]]        # 不持有/持有
        for i in range(1, n):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i] - fee)
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])
        return dp[-1][0]

    """ ------------------- 1个数组 最长递增子序列 连续or不连续 -------------------- 
        dp[i] 以nums[i]为结尾的... """
    def lengthOfLIS(self, nums: List[int]) -> int:
        """ 300.最长递增子序列
            递推公式不好想 dp[i] = max(dp[i], dp[j] + 1) """
        n = len(nums)
        dp = [1] * n        # dp[i] 以nums[i]结尾的最长递增子序列长度
        # 计算dp[i]时，从0~i-1遍历找加入nums[i]后能更长的
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)

    def findLengthOfLCIS(self, nums: List[int]) -> int:
        """ 674.最长连续递增序列
            那就看nums[i]能不能放到nums[i-1]后面呗 """
        n = len(nums)
        dp = [1] * n        # dp[i] 以nums[i]结尾的最长连续递增子序列的长度
        for i in range(1, n):
            if nums[i - 1] < nums[i]:
                dp[i] = dp[i - 1] + 1
        return max(dp)

    """ ------------------- 2个数组 最长子序列 连续or不连续 -------------------- 
        dp[i][j] 以nums1[i-1]、nums2[j-1]结尾的... """
    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        """ 718.最长重复子数组 连续
            '子数组'->原来本题也是求 连续子序列长度 """
        m, n = len(nums1), len(nums2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]        # dp[i][j] 以nums1[i-1]、nums[j-1]结尾的最长重复子数组的长度
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if nums1[i - 1] == nums2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
        return max(max(ls) for ls in dp)

    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        """ 1143.最长公共子序列 不连续：连续、不连续的结果都要 """
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]      # dp[i][j] text1的前i个元素、text2的前j个元素 组成的 最长公共子序列 长度
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])
        return max(max(ls) for ls in dp)

    def maxUncrossedLines(self, nums1: List[int], nums2: List[int]) -> int:
        """ 1035.不相交的线 不连续 """
        m, n = len(nums1), len(nums2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]      # dp[i][j] nums1的前i个元素、nums2的前j个元素 组成的 最长公共子序列(不连续) 长度
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if nums1[i - 1] == nums2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])
        return max(max(ls) for ls in dp)

    """ ------------------- 1个数组 最大子数组和 连续 -------------------- 
        dp[i] 以nums[i]为结尾的...  """
    def maxSubArray(self, nums: List[int]) -> int:
        """ 53.最大子数组和 连续 """
        n = len(nums)
        dp = [0] * n        # dp[i] 以nums[i]为结尾的 最大子数组和
        dp[0] = nums[0]
        for i in range(1, n):
            dp[i] = max(dp[i - 1] + nums[i], nums[i])
        return max(dp)

    """ ------------------- 2个数组 判断是否为子序列 -------------------- 
        dp[i][j] 以nums1[i-1]、nums2[j-1]为结尾的... """
    def isSubsequence(self, s: str, t: str) -> bool:
        """ 392.判断子序列 """
        m, n = len(s), len(t)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s[i - 1] == t[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    # 因为题目是判断s是否为t的子序列，所以s的全部元素都要在t中出现；只删t的元素就行，懂了不
                    # dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])
                    dp[i][j] = dp[i][j - 1]
        return dp[-1][-1] == m

    def numDistinct(self, s: str, t: str) -> int:
        """ 115.不同的子序列 """
        m, n = len(s), len(t)
        dp = [[0] * (n + 1) for _ in range(m + 1)]      # dp[i][j] 以s[i-1]为结尾的子串中 出现 以t[j-1]为结尾的子串 的次数
        # 初始化
        # dp[i][0]=1    以s[i-1]为结尾的子串 出现 空串 的次数 1次
        # dp[0][j]=0    空串 出现 以t[j-1]为结尾的子串 的次数 0次
        # dp[0][0]=1    空串 出现 空串 的次数 1次
        for i in range(1, m + 1):
            dp[i][0] = 1
        dp[0][0] = 1
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s[i - 1] == t[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j]
                else:
                    dp[i][j] = dp[i - 1][j]
        return dp[-1][-1]

    def minDistance(self, word1: str, word2: str) -> int:
        """ 583.两个字符串的删除操作 """
        # 自己写的，牛逼
        # m, n = len(word1), len(word2)
        # dp = [[0] * (n + 1) for _ in range(m + 1)]      # dp[i][j] 以word1[i-1]为结尾的子串 与 以word2[j-1]为结尾的子串 共需dp[i][j]个删除步骤
        # # 初始化
        # for i in range(1, m + 1):
        #     dp[i][0] = i
        # for j in range(1, n + 1):
        #     dp[0][j] = j
        # for i in range(1, m + 1):
        #     for j in range(1, n + 1):
        #         if word1[i - 1] == word2[j - 1]:
        #             dp[i][j] = dp[i - 1][j - 1]
        #         else:
        #             dp[i][j] = min(dp[i][j - 1] + 1, dp[i - 1][j] + 1)
        # return dp[-1][-1]

        # 曲线救国 求 最长公共子序列：连续、不连续都要
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]      # dp[i][j] 以word1[i-1]为结尾的子串、以word2[j-1]为结尾的子串 的最长公共子序列长度
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])
        return m + n - 2 * max(max(ls) for ls in dp)

    def minDistance(self, word1: str, word2: str) -> int:
        """ 72.编辑距离 """
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]      # dp[i][j] 以word1[i-1]为结尾的子串、以word2[j-1]为结尾的子串 所需最小操作步数
        # 初始化
        for i in range(1, m + 1):
            dp[i][0] = i
        for j in range(1, n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i][j - 1] + 1, dp[i - 1][j] + 1, dp[i - 1][j - 1] + 1)
        return dp[-1][-1]

    """ ------------------- 1个数组 另类题 连续or不连续 -------------------- """
    def countSubstrings(self, s: str) -> int:
        """ 647.回文子串 连续
            1个数组，dp定义与之前完全不同 """
        n = len(s)
        dp = [[False] * n for _ in range(n)]        # dp[i][j] 闭区间[i,j]是否回文
        # 初始化
        # for i in range(n):
        #     dp[i][i] = True
        for i in range(n - 1, -1, -1):
            for j in range(i, n):                   # 闭区间[i,j]嘛，所以j>=i
                if s[i] == s[j]:
                    if i == j:
                        dp[i][j] = True
                    elif abs(i - j) == 1:
                        dp[i][j] = True
                    elif dp[i + 1][j - 1]:
                        dp[i][j] = True
        return sum(sum(ls) for ls in dp)

    def longestPalindromeSubseq(self, s: str) -> int:
        """ 516.最长回文子序列 不连续 """
        n = len(s)
        dp = [[0] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                if s[i] == s[j]:
                    if i == j:
                        dp[i][j] = 1
                    elif abs(i - j) == 1:
                        dp[i][j] = 2
                    elif dp[i + 1][j - 1]:
                        dp[i][j] = dp[i + 1][j - 1] + 2
                else:
                    dp[i][j] = max(dp[i][j - 1], dp[i + 1][j])
        return max(max(ls) for ls in dp)

if __name__ == '__main__':
    sl = Solution()

    nums = [1, 1, 1, 1, 1]
    target = 3
    print(sl.test_multi_pack())

"""
动归五部曲：
    1.确定dp数组及下标的意义
    2.递推公式
    3.初始化dp
    4.确定遍历顺序
    5.手动模拟dp
"""