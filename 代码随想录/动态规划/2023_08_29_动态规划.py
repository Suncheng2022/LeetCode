from typing import List, Optional


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def fib(self, n: int) -> int:
        """ 509.斐波那契数 """
        if n <= 1:
            return n
        dp = [0] * (n + 1)
        dp[0], dp[1] = 0, 1
        for i in range(2, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[n]

    def climbStairs(self, n: int) -> int:
        """ 70.爬楼梯 """
        if n <= 1:
            return n
        dp = [0] * (n + 1)
        dp[1], dp[2] = 1, 2  # 爬到索引i楼梯，有dp[i]种方法; 不使用dp[0]，因为dp[0]如何初始化原因解释不通
        for i in range(3, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[n]

        # 优化一下空间复杂度
        # if n <= 1:
        #     return n
        # dp = [1, 2]
        # for i in range(3, n + 1):
        #     sum = dp[0] + dp[1]
        #     dp[0] = dp[1]
        #     dp[1] = sum
        # return dp[1]

    def minCostClimbingStairs(self, cost: List[int]) -> int:
        """ 746.使用最小花费爬楼梯 """
        dp = [0] * (len(cost) + 1)  # dp的含义要明确，到达下标i台阶所需要的最小花费为dp[i]
        dp[0], dp[1] = 0, 0  # 题目明确说了，可以选择从下标为0或1的台阶开始爬，即到达下标0或1的台阶不花费，继续往上爬就要花费体力
        for i in range(2, len(cost) + 1):
            # 状态转移公式：dp[i-1]转移到dp[i]——即从下标i-1继续往上爬就需要花费体力了，体力就是cost[i-1]；下标i-2往上爬类似
            dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])
        return dp[-1]

        # 优化 空间复杂度
        # dp = [0] * 2
        # dp[0], dp[1] = 0, 0     # 起点是下标0或1，到达起点不花费，继续往上爬才需要花费体力
        # for i in range(2, len(cost) + 1):
        #     dp_i = min(dp[0] + cost[i - 2], dp[1] + cost[i - 1])    # 有点绕，想一下索引i=2如何得到吧
        #     dp[0] = dp[1]
        #     dp[1] = dp_i
        # return dp[1]

    def uniquePaths(self, m: int, n: int) -> int:
        """ 62.不同路径 """
        dp = [[1] * n] + [[1] + [0] * (n - 1) for _ in range(m - 1)]  # 从位置[0,0]到达[i,j]的不同路径数
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[-1][-1]

    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        """ 63.不同路径II """
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        dp = [[0] * n for _ in range(m)]
        # 初始化dp首行
        for i in range(n):
            if obstacleGrid[0][i] == 1:
                break
            dp[0][i] = 1
        # 初始化dp首列
        for i in range(m):
            if obstacleGrid[i][0] == 1:
                break
            dp[i][0] = 1
        # print(dp)
        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] == 0:
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[-1][-1]

    def integerBreak(self, n: int) -> int:
        """ 343.整数拆分 """
        # 重写一遍
        dp = [0] * (n + 1)  # 拆分i得到的最大乘积为dp[i]
        dp[2] = 1  # 初始化dp，不使用dp[0]、dp[1] 因为不符合dp定义，直接从dp[2]开始计算
        for i in range(3, n + 1):
            for j in range(1, i):  # 其实j可以拆分到 n/2,因为拆分成近似大小的数字乘积才最大，拆分成n/2也就是拆分成两个数，两个数差不多都是n/2的情况
                dp[i] = max(dp[i], (i - j) * j, dp[i - j] * j)  # dp递推公式
        return dp[-1]

        # dp = [0] * (n + 1)
        # dp[2] = 1       # dp计算过程忽略dp[0]、dp[1]，因为这两者不符合dp含义；所以从dp[2]开始
        # for i in range(3, n + 1):
        #     for j in range(1, i):
        #         dp[i] = max(dp[i], (i - j) * j, dp[i - j] * j)      # dp[i]会通过for j循环计算多次，保留最大的而已
        # return dp[-1]

    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        """ 63.不同路径II """
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        dp = [[0] * n for _ in range(m)]
        # 初始化dp首行
        for i in range(n):
            if obstacleGrid[0][i] == 1:
                break
            dp[0][i] = 1
        # 初始化dp首列
        for i in range(m):
            if obstacleGrid[i][0] == 1:
                break
            dp[i][0] = 1
        # print(dp)
        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] == 0:
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[-1][-1]

    def integerBreak(self, n: int) -> int:
        """ 343.整数拆分 """
        # 重写一遍
        dp = [0] * (n + 1)  # 拆分i得到的最大乘积为dp[i]
        dp[2] = 1  # 初始化dp，不使用dp[0]、dp[1] 因为不符合dp定义，直接从dp[2]开始计算
        for i in range(3, n + 1):
            for j in range(1, i):  # 其实j可以拆分到 n/2,因为拆分成近似大小的数字乘积才最大，拆分成n/2也就是拆分成两个数，两个数差不多都是n/2的情况
                dp[i] = max(dp[i], (i - j) * j, dp[i - j] * j)  # dp递推公式
        return dp[-1]

        # dp = [0] * (n + 1)
        # dp[2] = 1       # dp计算过程忽略dp[0]、dp[1]，因为这两者不符合dp含义；所以从dp[2]开始
        # for i in range(3, n + 1):
        #     for j in range(1, i):
        #         dp[i] = max(dp[i], (i - j) * j, dp[i - j] * j)      # dp[i]会通过for j循环计算多次，保留最大的而已
        # return dp[-1]

    def numTrees(self, n: int) -> int:
        """ 96.不同的二叉搜索树 """
        dp = [0] * (n + 1)  # 1~i组成的节点，能组成BST的数量
        dp[0] = 1  # n为0的时候只有一种可能——空树
        for i in range(1, n + 1):
            for j in range(0, i):  # 其中一棵子树由j个节点组成 0<=j<=i-1, 因为根节点需要耗费1个节点
                dp[i] += dp[j] * dp[i - 1 - j]
        return dp[-1]

    def test_2_wei_bag_problem1(self):
        """ 01背包问题 导读1
            代码作参考而已 """
        weights = [1, 3, 4]
        values = [15, 20, 30]
        bagweight = 4

        # 初始化dp
        # 【注意】dp纵向表示可选择物品的索引0~i，注意0也表示物品，而不是不选；dp横向表示背包容量0~bagweight，注意有0
        # dp = [[0] * (bagweight + 1) for _ in range(len(weights))]
        # for j in range(1, len(dp[0])):
        #     if j >= weights[0]:
        #         dp[0][j] = values[0]
        # print(dp)
        #
        # # 遍历：先遍历物品 或 先遍历背包 都可以，看哪个好理解
        # for i in range(1, len(weights)):
        #     for j in range(1, bagweight + 1):
        #         # 物品i 放进去 or 没放进去，哪个价值更大
        #         dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weights[i]] + values[i])
        # print(dp[-1][-1])

        # 重写一遍
        # dp[i][j] 从0~i物品中选，背包大小为j，所能装下的最大价值
        dp = [[0] * (bagweight + 1) for _ in range(len(weights))]
        # 初始化 要同时考虑递推公式，思考需要先初始化哪些-->通常是由大约左上角元素推当前元素
        # j=0时肯定dp=0；i=0时dp首行能放下weight[0]的位置都初始化为weight[0]
        for j in range(weights[0], bagweight + 1):
            dp[0][j] = values[0]

        # 遍历
        for i in range(1, len(weights)):
            for j in range(1, bagweight + 1):
                if j < weights[i]:  # 放不下物品i
                    dp[i][j] = dp[i - 1][j]
                else:  # 放得下物品i
                    dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weights[i]] + values[i])
        print(dp[-1][-1])

    def test_1_wei_bag_problem(self):
        """ 01背包问题 导读2
            代码仅作参考 """
        weights = [1, 3, 4]
        values = [15, 20, 30]
        bagweight = 4

        # dp = [0] * (bagweight + 1)
        # for i in range(len(weights)):
        #     for j in range(bagweight, weights[i] - 1, -1):
        #         dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
        # return dp[-1]

        # 重写一遍
        # 一维dp[j]  表示背包容量j时，能装下的最大价值
        # 初始化 dp[0]=0  都放不下任何物品，价值自然是0
        dp = [0] * (bagweight + 1)

        for i in range(len(weights)):
            for j in range(bagweight, weights[i] - 1, -1):  # 隐含说明了 j是能放下weight[i]的
                dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
        return dp[-1]

    def canPartition(self, nums: List[int]) -> bool:
        """ 416.分割等和子集 """
        V = 100 * 200 // 2 + 1  # 由题意得 背包最大容量
        dp = [0] * V
        if sum(nums) % 2:
            return False
        target = sum(nums) // 2
        for i in range(len(nums)):
            for j in range(target, nums[i] - 1, -1):
                dp[j] = max(dp[j], dp[j - nums[i]] + nums[i])
        return dp[target] == target

    def lastStoneWeightII(self, stones: List[int]) -> int:
        """ 1049.最后一块石头的重量II """
        target = sum(stones) // 2  # 这里取整了，注意分成的2堆石头孰多孰少
        dp = [0] * (target + 1)  # 这里+1，否则报错越界；dp[i] 容量为i的背包 所能盛下最大重量(也说价值)为dp[i]
        # 一维01背包的遍历：外层是物品、内层是倒序遍历背包容量
        for i in range(len(stones)):
            for j in range(target, stones[i] - 1, -1):
                dp[j] = max(dp[j], dp[j - stones[i]] + stones[i])
        return (sum(stones) - dp[-1]) - dp[-1]  # 2堆石头相减，就是所求

    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        """ 494.目标和 """
        if (sum(nums) + target) % 2 or abs(target) > sum(nums):  # 无解
            return 0
        bagSize = (sum(nums) + target) // 2
        dp = [0] * (bagSize + 1)  # dp[i] 装满容量为i的背包，有dp[i]种方式
        dp[0] = 1  # 初始化
        for i in range(len(nums)):
            for j in range(bagSize, nums[i] - 1, -1):
                dp[j] += dp[j - nums[i]]
        return dp[-1]

    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        """ 474.一和零 """
        from collections import Counter

        dp = [[0] * (n + 1) for _ in range(m + 1)]  # dp[i][j] 最多有i个0、j个1 的 最大子集长度
        for s in strs:
            zeroNum = Counter(s)['0']
            oneNum = Counter(s)['1']
            for i in range(m, zeroNum - 1, -1):
                for j in range(n, oneNum - 1, -1):
                    dp[i][j] = max(dp[i][j], dp[i - zeroNum][j - oneNum] + 1)  # +1 可能表示比上一个dp多1个字符串
        return dp[-1][-1]

        # 重写一遍
        # dp = [[0] * (n + 1) for _ in range(m + 1)]      # 最多有i个0、j个1的  strs的最大子集的  大小为dp[i][j]--即字符串数量
        # for s in strs:
        #     zeroNum = Counter(s)['0']
        #     oneNum = len(s) - zeroNum
        #     for i in range(m, zeroNum - 1, -1):
        #         for j in range(n, oneNum - 1, -1):
        #             dp[i][j] = max(dp[i][j], dp[i - zeroNum][j - oneNum] + 1)
        # return dp[-1][-1]

    def test_CompletePack(self):
        """ 完全背包 导读 """
        # 已知条件
        weights = [1, 3, 4]
        vaules = [15, 20, 30]
        bagSize = 4
        # 初始化dp
        dp = [0] * (bagSize + 1)
        # 遍历计算
        for i in range(len(weights)):
            for j in range(weights[i], bagSize + 1):  # 完全背包，背包容量 则从小到大遍历
                dp[j] = max(dp[j], dp[j - weights[i]] + vaules[i])
        print(dp[-1])

    def change(self, amount: int, coins: List[int]) -> int:
        """ 518.零钱兑换II
            求组合数/装满背包有多少种方法，使用累加，初始化dp[0]=1 """
        dp = [0] * (amount + 1)  # dp[i] 容量为i的组合数是dp[i]
        dp[0] = 1
        for i in range(len(coins)):  # 遍历物品
            for j in range(coins[i], amount + 1):  # 遍历容量
                dp[j] += dp[j - coins[i]]
        return dp[-1]

    def combinationSum4(self, nums: List[int], target: int) -> int:
        """ 377.组合总和IV
            不限制使用元素个数，因此是完全背包问题
            本题求的是 排列，因此先遍历容量、再遍历物品；求组合反是 """
        dp = [0] * (target + 1)
        dp[0] = 1
        for i in range(target + 1):  # 遍历容量
            for j in range(len(nums)):  # 遍历物品
                if i >= nums[j]:  # 当背包>=当前物品的时候
                    dp[i] += dp[i - nums[j]]
        return dp[-1]

    def climbStairs(self, n: int) -> int:
        """ 70.爬楼梯(进阶版)
            之前使用回溯，直接 斐波那契数列
            本次使用动态规划，完全背包思路
            1.每次可以爬几层台阶--物品  n阶到达楼顶--背包容量
            2.物品可重复使用，完全背包 """
        weights = [1, 2]  # 物品
        dp = [0] * (n + 1)
        dp[0] = 1  # 完全背包 dp[0]初始化为1
        for i in range(n + 1):
            for j in range(len(weights)):  # [1,2]即物品
                if i >= weights[j]:
                    dp[i] += dp[i - weights[j]]
        return dp[-1]

    def coinChange(self, coins: List[int], amount: int) -> int:
        """ 322.零钱兑换
            金币可重复使用--完全背包
            1.不在乎顺序，所以物品、容量的先后遍历顺序均可，我们按组合思路做--先遍历物品、再遍历容量
            2.完全背包，内层循环正序 """
        dp = [float('inf')] * (amount + 1)  # dp[i] 组成总和i所需最少硬币数为dp[i]；初始化为最大，因为求最小值
        dp[0] = 0  # 强调一下这里的初始化 组成总和为0最少需要0个硬币
        for i in range(len(coins)):
            for j in range(amount + 1):
                if j >= coins[i]:  # 似乎是完全背包都有的if条件
                    dp[j] = min(dp[j], dp[j - coins[i]] + 1)
        return dp[-1] if dp[-1] != float('inf') else -1

    def numSquares(self, n: int) -> int:
        """ 279.完全平方数
            1.物品--完全平方数，背包--整数n
            2.物品可重复使用，完全背包 """
        dp = [float('inf')] * (n + 1)  # dp[i] 组成和为i的数 所需 完全平方数的最少个数；因为是求最小，所以初始化为最大值，从后面递推公式也可得出要初始化为最大值
        dp[0] = 0
        # 完全背包 则内外循环均可
        for i in range(1, int(pow(n, 0.5)) + 1):  # 物品
            for j in range(i ** 2, n + 1):  # 容量；容量范围稍加考虑，可省去if条件
                dp[j] = min(dp[j], dp[j - i ** 2] + 1)
        return dp[-1] if dp[-1] != float('inf') else 0

    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        """ 139.单词拆分
            之前用回溯递归，本次使用背包
            物品--字典的单词  背包--字符串s
            物品可重复使用--完全背包
            单词组成字符串包含‘顺序’，因此是求排列而非组合，因此外层遍历背包、内层遍历物品 """
        dp = [False] * (len(s) + 1)
        dp[0] = True
        # 两个for循环和答案略有出入，但也能通过，也能想明白
        for i in range(len(s) + 1):
            for j in range(i, len(s) + 1):
                if dp[i] and s[i:j] in wordDict:
                    dp[j] = True
        return dp[-1]

    def rob(self, nums: List[int]) -> int:
        """ 198.打家劫舍 """
        if len(nums) <= 2:
            return max(nums)
        dp = [0] * len(nums)  # dp[i] 截止到索引第i间房(包括) 能偷到的最大金额
        dp[0] = nums[0]
        dp[1] = max(nums[:2])
        for i in range(2, len(nums)):
            dp[i] = max(dp[i - 2] + nums[i], dp[i - 1])
        return dp[-1]

    def robII(self, nums: List[int]) -> int:
        """ 213.打家劫舍II """
        def rob_(nums, start, end):
            """ 即 198.打家劫舍 的逻辑 """
            dp = [0] * len(nums)
            dp[start] = nums[start]
            dp[start + 1] = max(nums[start: start + 2])
            for i in range(start + 2, end + 1):
                dp[i] = max(dp[i - 2] + nums[i], dp[i - 1])
            return dp[end]      # 注意返回dp的索引

        # robII的主逻辑
        if len(nums) == 2:
            return max(nums)
        elif len(nums) == 1:
            return nums[0]
        res1 = rob_(nums, 0, len(nums) - 2)
        res2 = rob_(nums, 1, len(nums) - 1)
        return max(res1, res2)

    def robIII(self, root: Optional[TreeNode]) -> int:
        """ 337.打家劫舍III
            '树形DP' 动态规划 """
        def robTree(node):
            """ 使用后序遍历，因为要通过递归函数的返回值做下一步计算 """
            if not node:
                return [0, 0]   # 不偷/偷 当前节点
            # 先处理左、右节点
            lefts = robTree(node.left)
            rights = robTree(node.right)
            # 再处理 中 节点
            # 不偷 or 偷
            val0 = max(lefts) + max(rights)
            val1 = node.val + lefts[0] + rights[0]
            return (val0, val1)
        return max(robTree(root))

    def maxProfit(self, prices: List[int]) -> int:
        """ 121.买卖股票的最佳时机
            本题‘只买卖一次’
            递推公式有些不明白 做完 122.买卖股票的最佳时机II 清楚很多！
            因为本题只能买卖一次，所以买入股票的时候不能考虑之前操作可能'剩了'多少钱，因为买意味着重新开始，不能考虑之前，否则不是一次买卖了 """
        # 动态规划 五部曲
        dp = [[0, 0] for _ in range(len(prices))]       # dp[i] 索引第i天的 持有/不持有 所获得的现金
        dp[0] = [-prices[0], 0]     # 初始化索引第0天的dp，持有只能是本天买入，不持有那就是不持有
        for i in range(1, len(prices)):
            dp[i][0] = max(dp[i - 1][0], -prices[i])
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] + prices[i])
        return dp[-1][1]

        # 本题是'简单'题，这个方法也简单，尝试动态规划
        # minVal = float('inf')       # 记录最小值
        # res = 0     # 保存最终结果
        # for p in prices:
        #     minVal = min(minVal, p)
        #     res = max(res, p - minVal)
        # return res

    def maxProfitII(self, prices: List[int]) -> int:
        """ 122.买卖股票的最佳时机II
            本题递推公式与 121.买卖股票的最佳时机 有差别，结合两道题目清楚很多 """
        dp = [[0, 0] for _ in range(len(prices))]
        dp[0] = [-prices[0], 0]
        for i in range(1, len(prices)):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] - prices[i])      # 本题允许多次买卖了，所以递推公式相比上一题也变化了
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] + prices[i])
        return dp[-1][1]

    def maxProfitIII(self, prices: List[int]) -> int:
        """ 123.买卖股票的最佳时机III
            最多可完成2笔交易
            虽然每天有5个状态，但是0状态不需要考虑，其他状态只由 前一天的2个状态转移而来，所以并不复杂
            状态:
                0 无操作
                1 第一次持有
                2 第一次不持有
                3 第二次持有
                4 第二次不持有 """
        dp = [[0] * 5 for _ in range(len(prices))]
        dp[0] = [0, -prices[0], 0, -prices[0], 0]
        for i in range(1, len(prices)):
            dp[i][1] = max(dp[i - 1][1], -prices[i])
            dp[i][2] = max(dp[i - 1][2], dp[i - 1][1] + prices[i])
            dp[i][3] = max(dp[i - 1][3], dp[i - 1][2] - prices[i])
            dp[i][4] = max(dp[i - 1][4], dp[i - 1][3] + prices[i])
        return dp[-1][-1]

    def maxProfitIV(self, k: int, prices: List[int]) -> int:
        """ 188.买卖股票的最佳时机IV
            相比上一题 123. 的至多2次交易，本题允许k次交易，依然延续上一题思路
            状态：
                0 无操作
                1 第一次持有
                2 第一次不持有
                3 第二次持有
                4 第二次不持有
                ...
                2*k+1 第k次不持有 """
        dp = [[0] * (2 * k + 1) for _ in range(len(prices))]
        for i in range(2 * k + 1):
            if i % 2:       # 初始化索引第0天 持有 的状态
                dp[0][i] = -prices[0]
        for i in range(1, len(prices)):
            for j in range(1, 2 * k + 1):
                if j % 2:   # 奇数 持有
                    dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - 1] - prices[i])
                else:       # 偶数 不持有
                    dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - 1] + prices[i])
        return dp[-1][-1]


if __name__ == '__main__':
    sl = Solution()

    k = 2
    prices = [3,2,6,5,0,3]
    print(sl.maxProfitIV(k, prices))

    """
    动态规划五部曲：
        1.确定dp数组以及下标的含义
        2.确定递推公式
        3.dp数组如何初始化
        4.确定遍历顺序
        5.举例推导dp数组
    """
