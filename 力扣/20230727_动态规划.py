from typing import List, Optional


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

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

    def change(self, amount: int, coins: List[int]) -> int:
        """ 518.零钱兑换II
            能想到背包问题--物品和容量 """
        dp = [0] * (amount + 1)
        dp[0] = 1
        for coin in coins:      # 本题求组成金额的硬币组合数，由示例可知是包含顺序的，因此从遍历硬币开始，对所有的金额i，其硬币组合顺序都有序--即硬币遍历顺序
            for i in range(coin, amount + 1):   # 遍历所有能允许加入硬币coin的金额
                dp[i] += dp[i - coin]       # 组成金额i-coin的，加上coin刚好组成金额i
        return dp[-1]

    def numSquares(self, n: int) -> int:
        """ 279.完全平方数
            dp[i]表示和为i的完全平方数的最少数量，对每个dp[i]遍历1~sqrt(i) """
        from math import sqrt

        dp = [0] * (n + 1)      # dp[0]为0，防止计算越界
        for i in range(1, n + 1):
            minn = n    # 记录和为i所需完全平方数的 最少数量
            for j in range(1, int(sqrt(i)) + 1):
                minn = min(minn, dp[i - j * j])     # 从 和i 中把j^2拿掉，用前面的已知推后面的
            dp[i] = minn + 1    # +1就是内层for去掉的某j^2
        return dp[-1]

    def rob(self, root: Optional[TreeNode]) -> int:
        """ 337.打家劫舍III
            直接看代码好懂些。递归回溯，是动态规划吗？"""
        def func(node):
            if not node:
                return [0, 0]
            lefts = func(node.left)
            rights = func(node.right)
            return [node.val + lefts[1] + rights[1],    # 偷当前节点
                    max(lefts) + max(rights)]           # 不偷当前节点

        return max(root.val + func(root.left)[1] + func(root.right)[1],
                   max(func(root.left)) + max(func(root.right)))

    def generateTrees(self, n: int) -> List[Optional[TreeNode]]:
        """ 95.不同的二叉搜索树II
            递归 回溯 """
        def func(start, end):
            """ 生成start~end能够成的所有子树，用于构建二叉搜索树 """
            if start > end:     # 无法构成二叉搜索树了
                return [None]   # None也是一种结果，比如有子树为空
            res = []
            for i in range(start, end + 1):     # start~end均要作为根节点
                lefts = func(start, i - 1)
                rights = func(i + 1, end)
                for l in lefts:
                    for r in rights:
                        root = TreeNode(i)
                        root.left = l
                        root.right = r
                        res.append(root)
            return res

        return func(1, n)

    def numTrees(self, n: int) -> int:
        """ 96.不同的二叉搜索树
            动态规划 不用回溯 https://leetcode.cn/problems/unique-binary-search-trees/solutions/330990/shou-hua-tu-jie-san-chong-xie-fa-dp-di-gui-ji-yi-h/?envType=featured-list&envId=2cktkvj"""
        dp = [0] * (n + 1)      # dp[i] i个节点能组成的BST数
        dp[0] = 1   # 0个节点只能组成1种--空树
        dp[1] = 1   # 1个节点只能组成1种--只有根节点的树
        for i in range(2, n + 1):
            for j in range(i):      # 其中一棵子树的节点数量为j，则另一棵为i-1-j
                dp[i] += dp[j] * dp[i - 1 - j]
        return dp[-1]

    def maxProfit(self, prices: List[int], fee: int) -> int:
        """ 714.买卖股票的最佳时机含手续费
            动态规划 2种状态：不持有、持有，用前一天推今天的结果 """
        n = len(prices)
        # 2种状态：1.不持有 2.持有
        dp = [[0, 0] for _ in range(n)]     # dp[i] 截止到索引第i天的 不持有 和 持有 的最大收益
        dp[0] = [0, -prices[0] - fee]
        for i in range(1, n):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i] - fee)
        return max([ls[0] for ls in dp])

    def maxProfit(self, prices: List[int]) -> int:
        """ 309.买卖股票的最佳时机含冷冻期
            股票买卖 每天几种状态
            据说官方答案解释很清楚 """
        n = len(prices)
        # 含冷冻期，则3种状态：1.持有 2.不持有，在冷冻期 3.不持有，不在冷冻期
        dp = [[0, 0, 0] for _ in range(n)]
        dp[0] = [-prices[0], 0, 0]      # 第1天/索引第0天要初始化，只能买才持有，不持有就不用区分了，因为这是第一天嘛
        for i in range(1, n):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][2] - prices[i])
            dp[i][1] = dp[i - 1][0] + prices[i]
            dp[i][2] = max(dp[i - 1][1], dp[i - 1][2])
        return max([max(ls[1:]) for ls in dp])

    def maxUncrossedLines(self, nums1: List[int], nums2: List[int]) -> int:
        """ 1035.不相交的线
            解题思路就是 1143.最长公共子序列 https://leetcode.cn/problems/uncrossed-lines/solutions/787955/bu-xiang-jiao-de-xian-by-leetcode-soluti-6tqz/?envType=study-plan-v2&envId=dynamic-programming"""
        m, n = len(nums1), len(nums2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]      # dp[i][j] nums1的前i个字符、nums2的前j个字符 的 最长公共子序列
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if nums1[i - 1] == nums2[j - 1]:    # i-1、j-1就刚好使nums1、nums2的索引不越界了
                    dp[i][j] = dp[i - 1][j - 1] + 1     # 【注意】dp索引与nums索引的意义区别
                else:
                    dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])
        return dp[-1][-1]

    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        """ 1143.最长公共子序列
            同 1035.不相交的线 """
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1     # +1就是指的if条件中相等的那个字符
                else:
                    dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])
        return dp[-1][-1]

    def longestArithSeqLength(self, nums: List[int]) -> int:
        """ 1027.最长等差数列
            固定等差d，动态规划 计算最长等差数列 没有使用dp """
        minV, maxV = min(nums), max(nums)
        difference = maxV - minV
        res = float('-inf')
        for d in range(-difference, difference + 1):    # 每固定“等差”，就遍历计算一次
            f = dict()      # f[k]=v 截止到元素k 最长等差数列的长度是v；没出现在f中，则当前遍历的n是等差数列的第一个元素
            for n in nums:
                if n - d in f:
                    f[n] = f[n - d] + 1
                else:
                    f[n] = 1
                res = max(res, f[n])
        return res

    def longestSubsequence(self, arr: List[int], difference: int) -> int:
        """ 1218.最长定差子序列
            同 1027.最长等差数列 """
        res = float('-inf')
        # for d in [-difference, difference]:     # 固定“等差”; 注意读题 多虑了
        f = dict()
        for n in arr:
            if n - difference in f:      # 当前遍历 n，若n-d访问过，则n不是子序列的首元素，使用前面的结果推即可
                f[n] = f[n - difference] + 1
            else:               # 当前遍历 n 是子序列首元素，长度置1，因为目前子序列只有n自己一个元素
                f[n] = 1
            res = max(res, f[n])
        return res

    def findLongestChain(self, pairs: List[List[int]]) -> int:
        """ 646.最长数对链
            题目说了 可以以任意顺序选择数对，那先排下序，方便计算当前遍历pairs[i]和之前pairs[0:i]比较大小
            https://leetcode.cn/problems/maximum-length-of-pair-chain/solutions/1793617/zui-chang-shu-dui-lian-by-leetcode-solut-ifpn/?envType=study-plan-v2&envId=dynamic-programming"""
        n = len(pairs)
        dp = [1] * n    # dp[i]截止到索引i个元素所组成的最长数对链；dp 必须初始化为1，不能初始化为0，因为每个单独的数对都是长度为1的数对链
        pairs.sort(key=lambda x: x[0])
        for i in range(1, n):
            for j in range(i):
                if pairs[j][-1] < pairs[i][0]:
                    dp[i] = max(dp[i], dp[j] + 1)   # dp[i]要保留内层for的最大值
        return dp[-1]

    def findNumberOfLIS(self, nums: List[int]) -> int:
        """ 673.最长递增子序列的个数
            dp记录截止到当前元素 最长递增子序列的 长度、cnt记录截止到当前元素 最长递增子序列 的长度出现的次数. 因为题目求的就是 出现次数 """
        # n = len(nums)
        # dp = [1] * n    # dp、cnt二者初始化为1，因为每个单独的元素也是 最长递增子序列 嘛
        # cnt = [1] * n
        # for i in range(n):
        #     for j in range(i):
        #         if nums[j] < nums[i]:
        #             if dp[i] < dp[j] + 1:
        #                 dp[i] = dp[j] + 1
        #                 cnt[i] = cnt[j]
        #             elif dp[i] == dp[j] + 1:
        #                 cnt[i] += cnt[j]
        # maxLen = max(dp)
        # return sum([cnt[i] for i, d in enumerate(dp) if d == maxLen])

        # 重写一遍
        # 题目没写可以任意顺序取 所以只能后面的元素和前面的依次比，必然有遍历
        n = len(nums)
        # dp、cnt均初始化为1, 因为每一个单独的元素 都是 递增子序列
        dp = [1] * n    # 截止到索引i元素可组成的递增子序列 长度
        cnt = [1] * n   # 截止到索引i元素可组成的递增子序列 长度 的出现次数，即组合方式数
        for i in range(n):      # 遍历计算dp[i]
            for j in range(i):  # 后面的元素 和 前面的元素 依次比，遍历
                if nums[j] < nums[i]:   # nums[i]是否能放到nums[j]后面
                    if dp[i] < dp[j] + 1:   # 放到nums[j]之后组成的递增序列长度可更新，同时要更新cnt[i]，因为最大长度变了 所以不是累加
                        dp[i] = dp[j] + 1
                        cnt[i] = cnt[j]
                    elif dp[i] == dp[j] + 1:    # 放到nums[j]之后递增序列长度不变，则不必更新dp[i]，但发现了组成长度dp[j]+1的其他方式，cnt累加
                        cnt[i] += cnt[j]
        maxLen = max(dp)    # 组成的 递增子序列 的最大长度
        return sum([cnt[i] for i, d in enumerate(dp) if d == maxLen])   # 返回所有 最大长度递增子序列 的出现次数之和，因为每个递增子序列最后一个元素是nums[i]，所以不会有重复

    def lengthOfLIS(self, nums: List[int]) -> int:
        """ 300.最长递增子序列
            同 673.最长递增子序列的个数 ，更简单 只求最长递增子序列的 长度 """
        n = len(nums)
        dp = [1] * n    # dp[i] 以nums[i]结尾 的 递增子序列的最大长度；初始化 每个单独的元素都是 递增子序列
        for i in range(n):      # 遍历计算dp[i]
            for j in range(i):  # nums[i]依次与nums[:i]比较，必然遍历
                if nums[j] < nums[i]:   # nums[i]能放到nums[j]后面
                    if dp[i] < dp[j] + 1:
                        dp[i] = dp[j] + 1
        return max(dp)

    def minimumDeleteSum(self, s1: str, s2: str) -> int:
        """ 712.两个字符串的最小ASCII删除和
            答案评论区 当成最长公共子序列：计算出公共子序列的ACSII和，然后与2字符串总和作差 """
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]      # dp[i][j] s1前i个字符 s2前j个字符 的 最长公共子序列的ACSII码之和
        sum = 0
        for c in s1:
            sum += ord(c)
        for c in s2:
            sum += ord(c)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + ord(s1[i - 1]) * 2     # 注意 *2；+1即if条件相等的字符
                else:
                    dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])      # 代码是不是很眼熟，以前是记录长度，现在记录ACSII码的和
        return sum - dp[-1][-1]


if __name__ == '__main__':
    sl = Solution()

    s1 = "delete"
    s2 = "leet"
    print(sl.minimumDeleteSum(s1, s2))
