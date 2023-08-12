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

    def longestPalindromeSubseq(self, s: str) -> int:
        """ 516.最长回文子序列
            寻找回文就要考虑两个端点 子序列则需要两个循环对吧 """
        n = len(s)
        dp = [[0] * n for _ in range(n)]    # dp[i][j] s的索引i~j范围内最长回文子序列的长度
        for i in range(n):      # 【注意】任意单独元素都是长度为1的 回文子序列
            dp[i][i] = 1
        for i in range(n - 1, -1, -1):      # i~j即寻找最长回文子序列的范围
            for j in range(i + 1, n):
                if s[i] == s[j]:
                    dp[i][j] = dp[i + 1][j - 1] + 2     # 首尾可以各增加1位
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])      # 首增加1尾 或 尾增加1位，取较大的
        return dp[0][-1]

    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        """ 139.单词拆分
            dp[i]表示前i位字符能否被字典表示 s[i:j]是否能表示 刚好能计算dp[j]"""
        n = len(s)
        dp = [False for _ in range(n + 1)]
        dp[0] = True    # 初始化dp 前0位认为能被字典表示
        for i in range(n + 1):   # 指示dp[i], 利用s[i:j]计算后面的dp[j]；注意，这里不是从1开始
            for j in range(i + 1, n + 1):
                if dp[i] and s[i:j] in wordDict:    # dp[i] 前i位、s[i:j] 二者刚号拼成s[:j]即dp[j]所表示的范围
                    dp[j] = True
        return dp[-1]

    def longestPalindrome(self, s: str) -> str:
        """ 5.最长回文子串
            题目意思似乎是找 索引连续的 子串 想到中心探测 """
        n = len(s)
        res = ""
        for i in range(n):
            l, r = i - 1, i + 1
            curRes = 1
            while l >= 0 and s[l] == s[i]:
                l -= 1
                curRes += 1
            # l += 1      # while结束，要么l越界、要么不等，所以右移一位到不越界且相等的位置
            while r <= n - 1 and s[r] == s[i]:
                r += 1
                curRes += 1
            # r -= 1
            while 0 <= l <= r <= n - 1 and s[l] == s[r]:
                l -= 1
                r += 1
                curRes += 2
            l += 1
            r -= 1
            res = s[l:r + 1] if r - l + 1 > len(res) else res
        return res

    def maximalSquare(self, matrix: List[List[str]]) -> int:
        """ 221.最大正方形
            官方答案最简单 dp大小同matrix
            https://leetcode.cn/problems/maximal-square/solutions/234964/zui-da-zheng-fang-xing-by-leetcode-solution/?envType=study-plan-v2&envId=dynamic-programming"""
        rows, cols = len(matrix), len(matrix[0])
        dp = [[0] * cols for _ in range(rows)]      # 官方答案 dp与matrix大小一致，省的考虑加一行一列了
        maxSide = 0
        for i in range(rows):
            for j in range(cols):
                if matrix[i][j] == '1':
                    if i == 0 or j == 0:
                        dp[i][j] = 1
                    else:
                        dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i][j - 1], dp[i - 1][j])
                maxSide = max(maxSide, dp[i][j])
        return maxSide ** 2

    def minFallingPathSum(self, matrix: List[List[int]]) -> int:
        """ 931.下降路径最小和
            动态规划 """
        rows, cols = len(matrix), len(matrix[0])
        for i in range(1, rows):
            for j in range(cols):
                curMin = matrix[i - 1][j]       # 注意 curMin的初始化 调试才知道
                if j - 1 >= 0:
                    curMin = min(curMin, matrix[i - 1][j - 1])
                if j + 1 <= cols - 1:
                    curMin = min(curMin, matrix[i - 1][j + 1])
                matrix[i][j] += curMin
        return min(matrix[-1])

    def minimumTotal(self, triangle: List[List[int]]) -> int:
        """ 120.三角形最小路径和
            动态规划 """
        for i in range(1, len(triangle)):
            for j in range(len(triangle[i])):
                # curMin的初始化要注意，要考虑别越了上一行的界
                if j <= len(triangle[i - 1]) - 1:
                    curMin = triangle[i - 1][j]
                else:
                    curMin = triangle[i - 1][j - 1]
                if j - 1 >= 0:
                    curMin = min(curMin, triangle[i - 1][j - 1])
                triangle[i][j] += curMin
        return min(triangle[-1])

    def maxCompatibilitySum(self, students: List[List[int]], mentors: List[List[int]]) -> int:
        """ 1947.最大兼容性评分和
            DETR 匈牙利算法 https://www.zhihu.com/tardis/zm/art/393028740?source_id=1003
            题目的意思是 计算 哪个学生与哪个老师的答案最接近 """
        # import numpy as np
        # from scipy.optimize import linear_sum_assignment
        #
        # all2all = []        # 所有学生对所有老师的匹配结果
        # for stu in students:
        #     stu2men = []    # 此学生对所有老师的匹配结果
        #     for men in mentors:
        #         curMatch = 0    # 遍历计算 此学生与1位老师的匹配结果
        #         for s, m in zip(stu, men):
        #             if s == m:  # 这个条件不能少，因为只有学生和老师相同题目答案相同才行
        #                 curMatch -= 1    # 这里也注意，不是减s+m；匈牙利算法的最优是最小的，所以我们指定“越匹配 则越小(匈牙利 小即是优)”
        #         stu2men.append(curMatch)
        #     all2all.append(stu2men)
        # print(f'{all2all}')
        # all2all = np.asarray(all2all)
        # rowInd, colInd = linear_sum_assignment(all2all)
        # return int(all2all[rowInd, colInd].sum()) * (-1)

        # 重写一遍
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        all2all = []    # 所有学生 对 所有老师 的匹配结果
        for stu in students:
            stu2mens = []       # 一个学生 对 所有老师 的匹配结果
            for men in mentors:
                curMatch = 0
                for s, m in zip(stu, men):
                    if s == m:
                        curMatch -= 1       # 只要 学生与老师 的 相同题目的答案相同，就“加1” 因为使用匈牙利算法，只能求最小方案，所以这里使用负数
                stu2mens.append(curMatch)
            all2all.append(stu2mens)

        all2all = np.asarray(all2all)
        rowInd, colInd = linear_sum_assignment(all2all)
        return int(all2all[rowInd, colInd].sum()) * (-1)

    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        """ 64.不同路径II """
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        dp = [[0] * n for _ in range(m)]
        dp[0][0] = 1    # 初始化配合后面判断 obstacleGrid[i][j]是否为1，才叫完美初始化
        for i in range(m):
            for j in range(n):
                if obstacleGrid[i][j] == 1:     # 遇到障碍物；同时能处理obstacleGrid只有1个元素的情况
                    dp[i][j] = 0
                    continue
                if i == 0 and j - 1 >= 0:
                    dp[i][j] = dp[i][j - 1]
                elif j == 0 and i - 1 >= 0:
                    dp[i][j] = dp[i - 1][j]
                elif i - 1 >= 0 and j - 1 >= 0:
                    dp[i][j] = (dp[i - 1][j] + dp[i][j - 1])
        return dp[-1][-1]

    def minPathSum(self, grid: List[List[int]]) -> int:
        """ 64.最小路径和"""
        m, n = len(grid), len(grid[0])
        for i in range(m):
            for j in range(n):
                if i == 0 and j - 1 >= 0:
                    grid[i][j] += grid[i][j - 1]
                elif j == 0 and i - 1 >= 0:
                    grid[i][j] += grid[i - 1][j]
                elif i - 1 >= 0 and j - 1 >= 0:
                    grid[i][j] += min(grid[i - 1][j], grid[i][j - 1])
        return grid[-1][-1]

    def uniquePaths(self, m: int, n: int) -> int:
        """ 62.不同路径 """
        dp = [[1] * n] + [[1] + [0] * (n - 1) for _ in range(m - 1)]        # 巧妙的初始化 首行首列为1
        print(dp)
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = (dp[i - 1][j] + dp[i][j - 1])
        return dp[-1][-1]

    def deleteAndEarn(self, nums: List[int]) -> int:
        """ 740.删除并获得点数
            类似 打家劫舍 的思路 """
        from collections import Counter

        num2count = Counter(nums)
        maxNum = max(nums)
        total = [0] * (maxNum + 1)      # 统计每个 num * counts
        for n, c in num2count.items():
            total[n] = n * c
        # 因为不能取连续的数字，所以接下来就是 打家劫舍 了
        n = len(total)
        dp = [0] * (n + 1)      # 前i个元素中，最多能偷多少钱
        dp[1] = total[0]
        for i in range(2, n + 1):       # 遍历计算dp
            dp[i] = max(dp[i - 2] + total[i - 1], dp[i - 1])        # 注意，当前元素是 total[i-1]
        return dp[-1]

    def rob(self, nums: List[int]) -> int:
        """ 198.打家劫舍
            不能偷相邻的屋子 """
        n = len(nums)
        dp = [0] * (n + 1)      # dp[i] 偷前i间屋子能获得的最大金额
        dp[1] = nums[0]
        for i in range(2, n + 1):
            dp[i] = max(dp[i - 2] + nums[i - 1], dp[i - 1])     # 注意，当前元素师 nums[i-1]
        return dp[-1]



if __name__ == '__main__':
    sl = Solution()

    nums = [2,2,3,3,3,4]
    print(sl.deleteAndEarn(nums))
