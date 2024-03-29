from typing import List, Optional


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def fib(self, n: int) -> int:
        """ 509.斐波那契数 """
        # dp = [0] * (n + 1)
        # if n < 1:                  # 处理边界
        #     return dp[n]
        # dp[0] = 0
        # dp[1] = 1
        # for i in range(2, n + 1):
        #     dp[i] = dp[i - 1] + dp[i - 2]
        # return dp[-1]

        # 优化一下空间复杂度
        if n <= 1:                  # 处理边界
            return n
        dp = [0] * 2                # dp初始化长度为2
        dp[0] = 0
        dp[1] = 1
        for i in range(2, n + 1):
            tmp = sum(dp)
            dp[0] = dp[1]
            dp[1] = tmp
        return dp[-1]

    def climbStairs(self, n: int) -> int:
        """ 70.爬楼梯
            递推公式 dp[i] = dp[i - 1] + dp[i - 2] """
        # if n == 1:
        #     return 1
        # dp = [0] * (n + 1)          # dp[i] 爬到第i阶楼梯有dp[i]种方法
        # dp[1] = 1                   # 初始化，不考虑dp[0]了，没实际意义，题目也已说明n>=1
        # dp[2] = 2
        # for i in range(3, n + 1):
        #     dp[i] = dp[i - 1] + dp[i - 2]
        # return dp[-1]

        # 优化空间复杂度
        # if n == 1:
        #     return 1
        # dp = [0] * 2                # 优化空间
        # dp[0] = 1
        # dp[1] = 1
        # for i in range(2, n + 1):
        #     tmp = sum(dp)
        #     dp[0] = dp[1]
        #     dp[1] = tmp
        # return dp[-1]

        # 或者可以这样优化空间
        if n == 1:
            return 1
        dp = [0] * 3
        dp[1] = 1
        dp[2] = 2
        for i in range(3, n + 1):
            tmp = dp[1] + dp[2]
            dp[1] = dp[2]
            dp[2] = tmp
        return dp[-1]

    def minCostClimbingStairs(self, cost: List[int]) -> int:
        """ 746.使用最小花费爬楼梯 """
        # 虽然自己写对了，还是看了一下《代码随想录》
        # n = len(cost)               # 一共多少阶楼梯
        # dp = [0] * (n + 1)          # dp[i] 爬到索引第i阶楼梯所需最小花费；得能爬到第n阶楼梯呀，所以索引要到n
        # dp[0] = 0                   # 初始化dp[0] dp[1]
        # dp[1] = 0
        # for i in range(2, n + 1):
        #     dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])       # 递推公式，到达第i阶 可以从第i-1阶、第i-2阶跳
        # return dp[-1]

        # 优化空间，虽然个人不太推荐，还是多会一个吧
        n = len(cost)
        dp = [0] * 2
        dp[0] = 0
        dp[1] = 0
        for i in range(2, n + 1):
            tmp = min(dp[1] + cost[i - 1], dp[0] + cost[i - 2])                   # 注意，dp长度为2了
            dp[0] = dp[1]
            dp[1] = tmp
        return dp[-1]

    def uniquePaths(self, m: int, n: int) -> int:
        """ 62.不同路径 """
        # dp = [[0] * n for _ in range(m)]            # dp[i][j] 到达坐标[i,j]有dp[i][j]种方法
        # for j in range(n):                          # 初始化 dp首行、首列为1
        #     dp[0][j] = 1
        # for i in range(m):
        #     dp[i][0] = 1
        # for i in range(1, m):
        #     for j in range(1, n):
        #         dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        # return dp[-1][-1]

        # 《代码随想录》稍微优化空间，还是建议上面写的，明了呀
        dp = [1] * n                        # 只使用1列，初始化为1
        for i in range(1, m):
            for j in range(1, n):
                dp[j] += dp[j - 1]          # 其实还是 当前=上+左
        return dp[-1]

    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        """ 63.不同路径II """
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        dp = [[0] * n for _ in range(m)]            # dp[i][j] 到达位置[i,j]有多少种方法
        for i in range(m):
            if obstacleGrid[i][0] == 1:             # 初始化注意，遇到'障碍物'就停止
                break
            dp[i][0] = 1
        for j in range(n):
            if obstacleGrid[0][j] == 1:
                break
            dp[0][j] = 1
        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] == 1:
                    continue
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[-1][-1]

    def integerBreak(self, n: int) -> int:
        """ 343.整数拆分
            注意递推公式，内层for多次计算同一个dp[i]，取最大的 """
        # 内层for范围不好确定，建议下面一种方法
        # from math import sqrt
        #
        # dp = [0] * (n + 1)                          # dp[i] 拆分整数i可获得的最大乘积
        # dp[2] = 1                                   # dp[0] dp[1]根本没法拆分，没意义
        # for i in range(3, n + 1):
        #     for j in range(1, int(sqrt(i)) + 2):    # int()是舍弃小数部分，也即向下取整，不会四舍五入；最安全的还是 i // 2 + 1
        #         dp[i] = max(dp[i], j * (i - j), j * dp[i - j])     # 拆分为2个、或2个以上，哪种方式乘积大
        # return dp[-1]

        # 《代码随想录》
        dp = [0] * (n + 1)                              # dp[i] 拆分i能得到的最大乘积
        dp[2] = 1                                       # dp[0]、dp[1]根本没意义
        for i in range(3, n + 1):
            for j in range(1, i // 2 + 1):              # 可以知道，一个数拆分为m个近似的数乘积最大，m至少得是2吧，所以j的范围 i // 2，range取到则 + 1
                dp[i] = max(dp[i], j * (i - j), j * dp[i - j])
        return dp[-1]

    def numTrees(self, n: int) -> int:
        """ 96.不同的二叉搜索树 """
        dp = [0] * (n + 1)                              # dp[i] i个节点能组成dp[i]个BST
        dp[0] = 1                                       # 纯为递推公式
        dp[1] = 1
        for i in range(2, n + 1):
            for j in range(i):                          # 其中一棵子树有j个节点，另一棵子树有i-1-j个节点，留下1个节点作父节点
                dp[i] += dp[j] * dp[i - 1 - j]
        return dp[-1]

    def test_2_wei_bag_problem1(self):
        """ 01背包导读 二维数组 """
        weights = [1, 3, 4]
        values = [15, 20, 30]
        bagweight = 4
        dp = [[0] * (bagweight + 1) for _ in range(len(weights))]       # m行n列 m:物品个数 n:背包容量
        for j in range(weights[0], bagweight + 1):                      # 首列初始化为0，首行看能否放下weights[0]
            dp[0][j] = values[0]
        for i in range(1, len(weights)):                                # 先遍历物品、再遍历容量 更好理解
            for j in range(weights[i], bagweight + 1):
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weights[i]] + values[i])
        return dp[-1][-1]       # 返回35

    def test_1_wei_bag_problem(self):
        """ 01背包 一维数组 """
        weights = [1, 3, 4]
        values = [15, 20, 30]
        bagweight = 4
        dp = [0] * (bagweight + 1)                  # 一维数组，去掉物品i的维度，只保留容量的维度
        for i in range(len(weights)):               # 一维数组，必须先遍历物品、再遍历背包，且内层倒序
            for j in range(bagweight, weights[i] - 1, - 1):
                dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
        return dp[-1]       # 35

    def canPartition(self, nums: List[int]) -> bool:
        """ 416.分割等和子集 """
        if sum(nums) % 2:
            return False
        target = sum(nums) // 2
        dp = [0] * (target + 1)         # dp[i] 容量为i能装下的物品最大的重量
        dp[0] = 0
        for i in range(len(nums)):
            for j in range(target, nums[i] - 1, -1):
                dp[j] = max(dp[j], dp[j - nums[i]] + nums[i])
        return dp[-1] == target

    def lastStoneWeightII(self, stones: List[int]) -> int:
        """ 1049.最后一块石头的重量II
            容量为sum(stones)//2的背包最多能放多少物品 """
        target = sum(stones) // 2
        dp = [0] * (target + 1)         # dp[i] 容量为i的背包能装下多大重量的物品
        dp[0] = 0
        for i in range(len(stones)):
            for j in range(target, stones[i] - 1, -1):
                dp[j] = max(dp[j], dp[j - stones[i]] + stones[i])
        return sum(stones) - 2 * dp[-1]

    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        """ 494.目标和
            组合问题 公式为dp[j] += dp[j-nums[i]]
            left - right = target
            left + right = sum
        得，left = (target + sum) / 2
        本题要明确dp的含义，dp[j] 容量为j的背包放满有dp[j]种方法，所以是 组合问题"""
        if (target + sum(nums)) % 2 or abs(target) > sum(nums):     # 边界条件注意
            return 0
        left = (target + sum(nums)) // 2
        dp = [0] * (left + 1)                   # dp[j] 装满容量为j的背包有dp[j]种方法
        dp[0] = 1                               # 一切递归的基础。也可理解为目标和为0时，只有一种方案，即什么都不选
        for i in range(len(nums)):
            for j in range(left, nums[i] - 1, -1):
                dp[j] += dp[j - nums[i]]        # 若找到nums[i]可以凑满dp[j]，则有dp[j-nums[i]]种方法，所有的方法加起来
        return dp[-1]

    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        """ 474.一和零 """
        from collections import Counter

        dp = [[0] * (n + 1) for _ in range(m + 1)]      # dp[i][j] 有 i个0、j个1 的最大子集长度是多少
        for i in range(len(strs)):
            zeroNum = Counter(strs[i])['0']
            oneNum = Counter(strs[i])['1']
            for j in range(m, zeroNum - 1, -1):
                for k in range(n, oneNum - 1, -1):
                    dp[j][k] = max(dp[j][k], dp[j - zeroNum][k - oneNum] + 1)
        return dp[-1][-1]

    def test_CompletePack(self):
        """ 完全背包 导读
            与 01背包 唯一的不同 物品数量不限制：
                1.遍历容量正序、倒序无所谓
                2.内外层物品、背包无所谓 """
        weights = [1, 3, 4]
        values = [15, 20, 30]
        bagWeight = 4
        dp = [0] * (bagWeight + 1)      # dp[j] 容量为j的背包所能盛下物品的最大价值
        dp[0] = 0                       # 也初始化一下吧
        for i in range(len(weights)):
            for j in range(weights[i], bagWeight + 1):
                dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
        print(dp[-1])                   # 60

    def change(self, amount: int, coins: List[int]) -> int:
        """ 518.零钱兑换II
            组合问题 dp[j] += dp[j-nums[i]]
            完全背包 内层for正序 """
        dp = [0] * (amount + 1)             # dp[i] 凑成金额i有几种方法
        dp[0] = 1                           # 似乎是递推的基础
        for i in range(len(coins)):
            for j in range(coins[i], amount + 1):
                dp[j] += dp[j - coins[i]]
        return dp[-1]

    def combinationSum4(self, nums: List[int], target: int) -> int:
        """ 377.组合总和IV
            排列问题：内外层遍历
            完全背包：物品无限，遍历物品正序，可重复使用 """
        dp = [0] * (target + 1)             # dp[i] 容量为i的背包装满有多少种方法
        dp[0] = 1
        for i in range(target + 1):         # 外层for容量、内层for物品，这样才能每个dp[i]考虑到不同物品的组合顺序，即体现排列思想
            for j in range(len(nums)):
                if i >= nums[j]:
                    dp[i] += dp[i - nums[j]]
        return dp[-1]

    def climbStairs(self, n: int) -> int:
        """ 70.爬楼梯
            使用背包思路
                完全背包：遍历容量正序
                排列问题：先遍历容量、再遍历物品 """
        dp = [0] * (n + 1)              # dp[i] 装满容量为i的背包有dp[i]种方法，组合问题不大对哈，排列问题就对味了
        dp[0] = 1                       # 递推的基础
        for i in range(1, n + 1):       # 先遍历背包
            for j in [1, 2]:            # 再遍历物品
                if i >= j:
                    dp[i] += dp[i - j]
        return dp[-1]

    def coinChange(self, coins: List[int], amount: int) -> int:
        """ 322.零钱兑换
            硬币无限--完全背包：遍历容量正序
        上来感觉不好想，动归五部曲响起：
            1.dp[i]意义
            2.递推公式 dp[j] = min(dp[j], dp[j - nums[i]] + 1)
            3.初始化 因为是求min，所以dp初始化为float('inf'), dp[0] = 0
            4.遍历顺序，完全背包 """
        # dp = [float('inf')] * (amount + 1)         # dp[i] 装满容量为i的背包所需硬币最少dp[i]个
        # dp[0] = 0
        # for i in range(len(coins)):
        #     for j in range(coins[i], amount + 1):
        #         dp[j] = min(dp[j], dp[j - coins[i]] + 1)
        # return dp[-1] if dp[-1] != float('inf') else -1

        # 《代码随想录》说本题与组合、排列没什么关系，所以遍历先后顺序无所谓
        dp = [float('inf')] * (amount + 1)          # 同上
        dp[0] = 0
        for i in range(1, amount + 1):
            for j in range(len(coins)):
                if i >= coins[j]:
                    dp[i] = min(dp[i], dp[i - coins[j]] + 1)
        return dp[-1] if dp[-1] != float('inf') else -1

    def numSquares(self, n: int) -> int:
        """ 279.完全平方数
            上来没思路，就想 动规五部曲：
                1.dp[i]意义
                2.递推公式 dp[i] = min(dp[i], dp[i - j ** 2] + 1)，稍微多思考一下就出来了
                3.初始化dp
                4.遍历顺序--完全背包
            判断好完全背包，下一步就要判断 组合/排列，进而确定遍历顺序 """
        # dp = [float('inf')] * (n + 1)              # dp[i] 和为i所需完全平方数的最少数量
        # dp[0] = 0
        # dp[1] = 1
        # for i in range(1, n // 2 + 1):
        #     for j in range(i * i, n + 1):
        #         dp[j] = min(dp[j], dp[j - i * i] + 1)
        # return dp[-1]

        # 理论上来讲，本题与组合/排列没什么关系，所以内外层for无所谓
        # 所以试试 先遍历容量、再遍历物品
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        dp[1] = 1
        for i in range(2, n + 1):               # 先遍历容量
            for j in range(1, n // 2 + 1):      # 再遍历物品
                if i >= j ** 2:
                    dp[i] = min(dp[i], dp[i - j ** 2] + 1)
        return dp[-1]

    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        """ 139.单词拆分
            单词不一定全使用，可重复使用 """
        n = len(s)
        dp = [False] * (n + 1)              # dp[i] 前i个字符能否被字典表示
        dp[0] = True
        for i in range(1, n + 1):           # 计算dp[i]
            for j in range(i):
                if dp[j] and s[j:i] in wordDict:
                    dp[i] = True
        return dp[-1]

    def rob(self, nums: List[int]) -> int:
        """ 198.打家劫舍 """
        if len(nums) == 1:
            return nums[0]
        n = len(nums)
        dp = [0] * n                  # dp[i] 截止到索引i家，所能偷得最大金额
        dp[0] = nums[0]
        dp[1] = max(nums[:2])
        for i in range(2, n):
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
        return dp[-1]

    def rob(self, nums: List[int]) -> int:
        """ 213.打家劫舍II
            成环 """
        def rob_(start, end):
            """ 其实就是 198.打家劫舍
                闭区间 """
            nums_cut = nums[start:end + 1]
            if len(nums_cut) == 1:
                return nums_cut[0]
            n = len(nums_cut)
            dp = [0] * n
            dp[0] = nums_cut[0]
            dp[1] = max(nums_cut[:2])
            for i in range(2, n):
                dp[i] = max(dp[i - 1], dp[i - 2] + nums_cut[i])
            return dp[-1]

        if len(nums) == 1:
            return nums[0]

        res1 = rob_(0, len(nums) - 2)
        res2 = rob_(1, len(nums) - 1)
        return max(res1, res2)

    def rob(self, root: Optional[TreeNode]) -> int:
        """ 337.打家劫舍III
            结合 回溯三部曲 """
        def backtracking(node):
            """ 偷 以node为根节点的子树 能获得的最大金额
                返回 [不偷node, 偷node] """
            # 终止条件
            if not node:
                return [0, 0]
            # 单层递归
            left = backtracking(node.left)
            right = backtracking(node.right)
            return [max(left) + max(right),
                    node.val + left[0] + right[0]]

        return max(backtracking(root))

    def maxProfit(self, prices: List[int]) -> int:
        """ 121.买卖股票的最佳时机
            一次买卖 """
        # 没有体现方法论
        # minPrice = float('inf')
        # for price in prices:
        #     if price < minPrice:
        #         minPrice = price
        #         res = price - minPrice
        # return res

        # 还是得《代码随想录》
        # 2种状态 不持有/持有
        n = len(prices)
        dp = [[0, 0] for _ in range(n)]         # dp[i][j] 第i天 不持有/持有 状态时，所获得最大利润
        dp[0] = [0, -prices[0]]
        for i in range(1, n):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])      # 计算 不持有
            dp[i][1] = max(dp[i - 1][1], -prices[i])                    # 计算 持有，注意 一次买卖
        return dp[-1][0]

    def maxProfit(self, prices: List[int]) -> int:
        """ 122.买卖股票的最佳时机II
            多次买卖 """
        n = len(prices)
        dp = [[0, 0] for _ in range(n)]         # dp[i][j] 第i天 不持有/持有 状态时，所获得最大利润
        dp[0] = [0, -prices[0]]
        for i in range(1, n):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])      # 计算 不持有
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])      # 计算 持有，注意 多次买卖 买之前有盈余了
        return dp[-1][0]

    def maxProfit(self, prices: List[int]) -> int:
        """ 123.买卖股票的最佳时机III
            两次买卖
                状态0：无操作
                状态1：第一次持有
                状态2：第一次不持有
                状态3：第二次持有
                状态4：第二次不持有 """
        n = len(prices)
        dp = [[0] * 5 for _ in range(n)]        # 5个状态
        dp[0] = [0, -prices[0], 0, -prices[0], 0]
        for i in range(1, n):
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])
            dp[i][2] = max(dp[i - 1][2], dp[i - 1][1] + prices[i])
            dp[i][3] = max(dp[i - 1][3], dp[i - 1][2] - prices[i])
            dp[i][4] = max(dp[i - 1][4], dp[i - 1][3] + prices[i])
        return max(dp[-1])

    def maxProfit(self, k: int, prices: List[int]) -> int:
        """ 188.买卖股票的最佳时机IV
            k次买卖 """
        n = len(prices)
        dp = [[0] * (2 * k + 1) for _ in range(n)]
        for i in range(2 * k + 1):
            if i % 2:                                   # 奇数 持有
                dp[0][i] = -prices[0]
        for i in range(1, n):
            for j in range(1, 2 * k + 1):
                if j % 2:                               # 计算 持有
                    dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - 1] - prices[i])
                else:                                   # 计算 不持有
                    dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - 1] + prices[i])
        return dp[-1][-1]

    def maxProfit(self, prices: List[int]) -> int:
        """ 309.买卖股票的最佳时机含冷冻期
            状态0：持有
            状态1：不持有，在冷冻期
            状态2：不持有 """
        n = len(prices)
        dp = [[0, 0, 0] for _ in range(n)]
        dp[0] = [-prices[0], 0, 0]
        for i in range(1, n):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][2] - prices[i])
            dp[i][1] = dp[i - 1][0] + prices[i]
            dp[i][2] = max(dp[i - 1][2], dp[i - 1][1])
        return max(dp[-1])

    def maxProfit(self, prices: List[int], fee: int) -> int:
        """ 714.买买股票的最佳时机含手续费
            多次买卖 """
        n = len(prices)
        dp = [[0, 0] for _ in range(n)]             # dp[i][j] 索引第i天j状态所获最大利润；[不持有，持有]
        dp[0] = [0, -prices[0]]
        for i in range(1, n):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i] - fee)
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])
        return dp[-1][0]

    def lengthOfLIS(self, nums: List[int]) -> int:
        """ 300.最长递增子序列 """
        n = len(nums)
        dp = [1] * n                                # 截止到索引第i个元素 的 最长递增子序列长度；长度至少是1，元素本身
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)

    def findLengthOfLCIS(self, nums: List[int]) -> int:
        """ 674.最长连续递增序列 """
        n = len(nums)
        dp = [1] * n                                # dp[i] 截止到索引第i个元素，最长连续递增子序列的长度
        for i in range(1, n):
            if nums[i - 1] < nums[i]:
                dp[i] = dp[i - 1] + 1
        return max(dp)

    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        """ 718.最长重复子数组 难
            必看《代码随想录》
            '子数组'意味着'连续' """
        m = len(nums1)
        n = len(nums2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]          # dp[i][j] 以nums1[i-1]为结尾、nums2[j-1]为结尾的 最长重复子数组的长度
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if nums1[i - 1] == nums2[j - 1]:            # 以nums1[i-1]、nums2[j-1]为结尾的子数组——即dp[i][j]的含义
                    dp[i][j] = dp[i - 1][j - 1] + 1
        return max([max(ls) for ls in dp])

    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        """ 1143.最长公共子序列
            '子序列'意味着'不连续' """
        m = len(text1)
        n = len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]          # dp[i][j] 以text1[i-1]、text2[j-1]结尾的 最长公共子序列长度
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return max([max(ls) for ls in dp])

    def maxUncrossedLines(self, nums1: List[int], nums2: List[int]) -> int:
        """ 1035.不相交的线
            其实就是 1143.最长公共子序列 """
        m = len(nums1)
        n = len(nums2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if nums1[i - 1] == nums2[j - 1]:            # 明确dp[i][j]意义
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return max([max(ls) for ls in dp])

    def maxSubArray(self, nums: List[int]) -> int:
        """ 53.最大子数组和
            '子数组'意味着'连续'
            注意：本题dp的定义 """
        n = len(nums)
        dp = [0] * n                                    # dp[i] 以nums[i]为结尾的 最大子数组和
        dp[0] = nums[0]
        for i in range(1, n):
            dp[i] = max(nums[i], dp[i - 1] + nums[i])     # 递推公式体现了'连续'
        return max(dp)

    def isSubsequence(self, s: str, t: str) -> bool:
        """ 392.判断子序列
            即 1143.最长公共子序列 """
        # m = len(s)
        # n = len(t)
        # dp = [[0] * (n + 1) for _ in range(m + 1)]      # dp[i][j] 以s[i-1]结尾、t[j-1]结尾的 最长公共子序列长度
        # for i in range(1, m + 1):
        #     for j in range(1, n + 1):
        #         if s[i - 1] == t[j - 1]:
        #             dp[i][j] = dp[i - 1][j - 1] + 1
        #         else:
        #             dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        # return dp[-1][-1] == m

        # 《代码随想录》讲的更精确、完美
        m = len(s)
        n = len(t)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s[i - 1] == t[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = dp[i][j - 1]             # 如果不等，删t的字母即可，因为题目是判断s是否为t的子序列
        return dp[-1][-1] == m

    def numDistinct(self, s: str, t: str) -> int:
        """ 115.不同的子序列 很难 """
        m = len(s)
        n = len(t)
        dp = [[0] * (n + 1) for _ in range(m + 1)]      # dp[i][j] 以t[j-1]结尾的子序列 在 以s[i-1]结尾的子序列 中出现的次数
        # 初始化dp
        for i in range(1, m + 1):
            dp[i][0] = 1                                # 根据dp定义dp[i][0]也是1；以s[i-1]结尾的子序列 删除所有元素即可出现一次 空
        for j in range(1, n + 1):
            dp[0][j] = 0                                # 空 无论如何也不会包含 不空，所以出现0次
        dp[0][0] = 1                                    # 空 出现 空，可以出现1次
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s[i - 1] == t[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j]
                else:
                    dp[i][j] = dp[i - 1][j]
        return dp[-1][-1]

    def minDistance(self, word1: str, word2: str) -> int:
        """ 583.两个字符串的删除操作
            115.不同的子序列 只能删一个字符串的字符，本题都能删 """
        m = len(word1)
        n = len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]      # dp[i][j] 以word1[i-1]结尾、以word2[j-1]结尾 的字符串，一共删除多少次能相同
        for i in range(1, m + 1):
            dp[i][0] = i
        for j in range(1, n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 2)
        return dp[-1][-1]

    def minDistance(self, word1: str, word2: str) -> int:
        """ 72.编辑距离 """
        m = len(word1)
        n = len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]      # dp[i][j] 从头截止到word1[i-1]结尾的子串 变成 从头截止到word2[j-1]结尾的子串 需要最少的操作数
        for i in range(1, m + 1):
            dp[i][0] = i
        for j in range(1, n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:        # 相等 不需要操作
                    dp[i][j] = dp[i - 1][j - 1]
                else:                                   # 不相等 增、删、替换
                    dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1)
        return dp[-1][-1]

    # ---------------- 重写 编辑距离 --------------------
    def isSubsequence(self, s: str, t: str) -> bool:
        """ 392.判断子序列
            判断s是否为t的子序列，不等的时候 只能删t """
        # 判断s是否为t的子序列
        m = len(s)
        n = len(t)
        dp = [[0] * (n + 1) for _ in range(m + 1)]    # dp[i][j] 从头开始到以s[i-1]结尾、从头开始到以t[j-1]结尾的 公共子序列长度
        # 初始化dp 思考dp定义，首行、首列都是0
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s[i - 1] == t[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = dp[i][j - 1]           # 判断s是否为t的子序列，所以只能删t
        return dp[-1][-1] == m

    def numDistinct(self, s: str, t: str) -> int:
        """ 115.不同的子序列
            s的子序列中 t出现的次数，所以只能删s
            等：用它来匹配、不用它来匹配
            不等：不用它来匹配 """
        m = len(s)
        n = len(t)
        dp = [[0] * (n + 1) for _ in range(m + 1)]          # dp[i][j] 从头到以s[i-1]结尾的子序列中 从头到t[j-1]结尾的子序列 出现的次数
        # 初始化，首列dp[i][0]=1，你想想是不；也可以这么想，删s能得到几次t
        # 初始化，首行dp[0][j]=0 s是空，再怎么删也得不到j的不空的子序列
        for i in range(m + 1):
            dp[i][0] = 1                                    # s子序列全删除，能到 空，所以1次
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s[i - 1] == t[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j]      # 相等，用/不用 s[i-1]来匹配
                else:
                    dp[i][j] = dp[i - 1][j]                         # 不相等   不用 s[i-1]来匹配
        return max([ls[-1] for ls in dp])

    def minDistance(self, word1: str, word2: str) -> int:
        """ 583.两个字符串的删除操作
            两个字符都可以删除 """
        m = len(word1)
        n = len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]          # 以word1[i-1]结尾 与 以word2[j-1]结尾 相等需要删除的次数
        # 初始化
        dp[0][0] = 0
        for i in range(1, m + 1):
            dp[i][0] = i
        for j in range(1, n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1)      # word1[i - 1] != word2[j - 1] dp[i - 1][j]或dp[i][j - 1]再删1个 得到dp[i-1][j-1]
        return dp[-1][-1]

    def minDistance(self, word1: str, word2: str) -> int:
        """ 72.编辑距离
            绝杀！ """
        m = len(word1)
        n = len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]          # 以word1[i-1]结尾、以word2[j-1]结尾 变为相同需要 增、删、改 总次数
        dp[0][0] = 0
        for i in range(1, m + 1):
            dp[i][0] = i
        for j in range(1, n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1)        # 最终的状态都是 word1[i - 1] == word2[j - 1] 嘛
        return dp[-1][-1]

    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """ 24.两两交换链表中的节点
            吉利研究院一面 拉垮呀，这题竟然没做出来
            精髓：cur迅速指向第一个未处理节点
                 pre迅速指向最后一个已处理节点 """
        # 本题还是比较难的，现场没有coding出来也正常
        if not head or not head.next:           # 边界条件，处理0或1个节点的情况
            return head
        cur = head
        stack = [cur, cur.next]
        cur = cur.next.next                     # 迅速指向 第一个尚未处理的节点

        head = stack.pop()
        head.next = stack.pop()
        pre = head.next                         # 迅速指向 最后一个已处理的节点

        while cur and cur.next:
            stack.extend([cur, cur.next])
            cur = cur.next.next

            pre.next = stack.pop()
            pre.next.next = stack.pop()
            pre = pre.next.next
        if cur and not cur.next:
            pre.next = cur
            pre = pre.next
        pre.next = None
        return head

    def countSubstrings(self, s: str) -> int:
        """ 647.回文子串
            见 《代码随想录》 """
        # n = len(s)
        # dp = [[0] * n for _ in range(n)]        # dp[i][j] 闭区间[i,j]是否回文
        # for i in range(n - 1, -1, -1):          # 根据递推公式，从下往上、从左往右遍历
        #     for j in range(i, n):
        #         if s[i] == s[j]:
        #             if j - i <= 1:
        #                 dp[i][j] = True
        #             elif dp[i + 1][j - 1]:
        #                 dp[i][j] = True
        # return sum([sum(ls) for ls in dp])

        # 双指针，当年做过，不过还是倾向于动态规划
        # def extend(i, j):
        #     res = 0
        #     while 0 <= i <= j < len(s) and s[i] == s[j]:
        #         res += 1
        #         i -= 1
        #         j += 1
        #     return res
        #
        # res = 0
        # for i in range(len(s)):
        #     res += extend(i, i)
        #     if i < len(s) - 1:
        #         res += extend(i, i + 1)
        # return res

        # 双指针 简化一下
        def extend(i, j):
            res = 0
            while 0 <= i <= j < len(s) and s[i] == s[j]:
                res += 1
                i -= 1
                j += 1
            return res

        res = 0
        for i in range(len(s)):
            res += extend(i, i)
            res += extend(i, i + 1)
        return res

    def longestPalindromeSubseq(self, s: str) -> int:
        """ 516.最长回文子序列
            子序列-不连续
            子串-连续 """
        n = len(s)
        dp = [[0] * n for _ in range(n)]      # dp[i][j] 闭区间[i,j] 最长回文子序列 的长度
        for i in range(n):
            dp[i][i] = 1
        for i in range(n - 1, -1, -1):
            for j in range(i + 1, n):         # 计算不到i==j的情况，所以跳过
                if s[i] == s[j]:              # 《代码随想录》所说 “从递推公式：dp[i][j] = dp[i + 1][j - 1] + 2; 可以看出 递推公式是计算不到 i 和j相同时候的情况”，因为只计算了dp右上三角，i==j时无法计算出正确结果，所以要提前初始化
                    dp[i][j] = dp[i + 1][j - 1] + 2
                else:
                    dp[i][j] = max(dp[i][j - 1], dp[i + 1][j])
        return max(max(ls) for ls in dp)


"""
动规五部曲：
    1.明确dp下标及dp[i]的含义
    2.确定递推公式
    3.初始化
    4.确定遍历顺序
    5.
"""
if __name__ == "__main__":
    sl = Solution()

    print(sl.test_CompletePack())