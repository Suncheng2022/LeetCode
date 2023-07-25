from typing import List, Optional


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def climbStairs(self, n: int) -> int:
        """
        70.爬楼梯
        2023.07.19 简单
        题解：斐波那契数列
        """
        dp = [0 for _ in range(n + 1)]
        dp[0] = 1
        dp[1] = 1
        for i in range(2, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[-1]

    def fib(self, n: int) -> int:
        """
        509.斐波那契数
        2023.07.19 简单
        """
        if n == 0:
            return 0
        dp = [0 for _ in range(n + 1)]
        dp[0] = 0
        dp[1] = 1
        for i in range(2, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[n]

    def tribonacci(self, n: int) -> int:
        """
        1137.第N个泰波那契数
        2023.07.19 简单
        """
        if n == 0:
            return 0
        elif n <= 2:
            return 1
        dp = [0 for _ in range(n + 1)]
        dp[0] = 0
        dp[1] = 1
        dp[2] = 1
        for i in range(3, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2] + dp[i - 3]
        return dp[-1]

    def minCostClimbingStairs(self, cost: List[int]) -> int:
        """
        746.使用最小花费爬楼梯
        2023.07.19 简单
        题解：
        """
        n = len(cost)
        dp = [0 for _ in range(n + 1)]      # 到达i时的最小花费(只计算到达i所经过的元素，不包括i)
        dp[0] = dp[1] = 0       # 只计算到达i所经过的元素，不包括i
        for i in range(2, n + 1):
            dp[i] = min(dp[i - 1] + cost[i - 1],    # 从i-1到达i，就要计算经过i-1的费用
                        dp[i - 2] + cost[i - 2])    # 从i-2到达i，就要计算经过i-2的费用
        return dp[n]

    def rob(self, nums: List[int]) -> int:
        """
        198.打家劫舍
        2023.07.19 中等
        """
        # 重写一遍
        n = len(nums)
        dp = [0 for _ in range(n + 1)]      # 前i间房屋能偷到的最大金额
        dp[1] = nums[0]
        for i in range(2, n + 1):       # 计算前i间房屋能偷到多少，注意与nums[i]的i区分
            dp[i] = max(nums[i - 1] + dp[i - 2], dp[i - 1])
        return dp[n]

        # n = len(nums)
        # dp = [0 for _ in range(n + 1)]
        # dp[1] = nums[0]
        # for i in range(2, n + 1):
        #     dp[i] = max(dp[i - 1], dp[i - 2] + nums[i - 1])
        # return dp[n]

    def deleteAndEarn(self, nums: List[int]) -> int:
        """
        740.删除并获得点数
        2023.07.21 中等
        题解：类似 198.打家劫舍
        """
        from collections import Counter

        num2counts_dict = Counter(nums)
        total = [0] * (max(nums) + 1)
        for num, counts in num2counts_dict.items():
            total[num] = num * counts
        # 后面和打家劫舍一样了
        dp = [0 for _ in range(len(total) + 1)]
        dp[1] = total[0]
        for i in range(2, len(total) + 1):
            dp[i] = max(total[i - 1] + dp[i - 2], dp[i - 1])
        return dp[-1]

    def uniquePaths(self, m: int, n: int) -> int:
        """
        62.不同路径
        2023.07.21 中等
        """
        dp = [[1] * n] + [[1] + [0] * (n - 1) for _ in range(m - 1)]
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[-1][-1]

    def minPathSum(self, grid: List[List[int]]) -> int:
        """
        64.最小路径和
        2023.07.21 中等
        """
        m = len(grid)
        n = len(grid[0])
        for i in range(m):
            for j in range(n):
                if i == 0 and j - 1 >= 0:
                    grid[i][j] += grid[i][j - 1]
                elif j == 0 and i - 1 >= 0:
                    grid[i][j] += grid[i - 1][j]
                elif i - 1 >= 0 and j - 1 >= 0:       # 这个条件注意，[0][0]就能走进来
                    grid[i][j] += min(grid[i][j - 1], grid[i - 1][j])
        return grid[-1][-1]

    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        """
        63.不同路径II
        2023.07.21 中等
        题解：看下之前记录的代码，思路没那么复杂哦
        """
        m = len(obstacleGrid)
        n = len(obstacleGrid[-1])
        dp = [[0] * n for _ in range(m)]
        dp[0][0] = 1
        for i in range(m):
            for j in range(n):
                if obstacleGrid[i][j]:      # 遍历到障碍物
                    dp[i][j] = 0            # 到障碍物网格路径数量为0
                    continue                # 这也不能忘，遇到障碍物也是处理了此处的dp
                if i == 0 and j - 1 >= 0:
                    dp[i][j] = dp[i][j - 1]
                elif i - 1 >= 0 and j == 0:
                    dp[i][j] = dp[i - 1][j]
                elif i - 1 >= 0 and j - 1 >= 0:
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[-1][-1]

    def minimumTotal(self, triangle: List[List[int]]) -> int:
        """
        120.三角形最小路径和
        2023.07.21 中等
        """
        # 自己改为了本地修改！牛逼啦！
        for i in range(1, len(triangle)):       # 三角形 第一行只有1个元素
            for j in range(len(triangle[i])):
                if j == 0:
                    triangle[i][j] += triangle[i - 1][j]
                elif j == len(triangle[i]) - 1:
                    triangle[i][j] += triangle[i - 1][-1]   # 此处两个j处都为-1
                else:
                    triangle[i][j] += min(triangle[i - 1][j - 1], triangle[i - 1][j])
        return min(triangle[-1])

        # dp = [[0] * len(ls) for ls in triangle]
        # dp[0][0] = triangle[0][0]
        # for i in range(1, len(triangle)):       # 猜测 三角形第一行只1个元素
        #     for j in range(len(triangle[i])):
        #         # 特殊处理当前处理行的 第一个 和 最后一个
        #         if j == len(triangle[i]) - 1:
        #             dp[i][j] = dp[i - 1][j - 1] + triangle[i][j]
        #         elif j - 1 >= 0:
        #             dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j]) + triangle[i][j]
        #         elif j - 1 < 0:     # j < 1
        #             dp[i][j] = dp[i - 1][0] + triangle[i][j]
        # return min(dp[-1])

    def minFallingPathSum(self, matrix: List[List[int]]) -> int:
        """
        931.下降路径最小和
        2023.07.21 中等
        题解：思路很像 120.三角形最小路径和，从上往下走；这里需要注意的是 取 当前元素 的上一行的最小值有些绕
        """
        n = len(matrix)
        dp = [[0] * n for _ in range(n)]
        dp[0] = matrix[0][:]    # 注意此处深拷贝，否则会指向同一块内存
        # 当前元素对应上一行的3个元素，取最小值；
        # 如何取最小值需要考虑得尽量鲁棒，不要想得太复杂，自己下面写的思路就很好，不但正确取了最小值，还完美处理了边界问题
        for i in range(1, n):
            for j in range(n):
                tmp_min = float('inf')
                # 注意 取最小值
                if j - 1 >= 0:      # 当前元素的列 往左边探一下是否超边界
                    tmp_min = min(dp[i - 1][j - 1], dp[i - 1][j])
                if j + 1 <= n - 1:  # 当前元素的列 往右边探一下是否超边界
                    tmp_min = min(dp[i - 1][j], dp[i - 1][j + 1]) if min(dp[i - 1][j], dp[i - 1][j + 1]) < tmp_min else tmp_min
                dp[i][j] = tmp_min + matrix[i][j]
        return min(dp[-1])

    def maximalSquare(self, matrix: List[List[str]]) -> int:
        """
        221.最大正方形
        2023.07.21 中等
        题解：还是参考答案解法--dp添加一行0、一列0，dp[i+1][j+1]表示matrix[i][j]为右下角时正方形最大边长，这样可以完整地遍历一遍matrix，最后不需要特殊处理了
        """
        # 答案解法
        m = len(matrix)
        n = len(matrix[0])
        dp = [[0] * (n + 1) for _ in range(m + 1)]      # dp添加一行0、一列0，可以完整遍历matrix
        # 开始遍历，索引表示matrix
        maxSide = 0
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == '1':
                    dp[i + 1][j + 1] = 1 + min(dp[i][j + 1], dp[i + 1][j], dp[i][j])
                    maxSide = max(maxSide, dp[i + 1][j + 1])
        return maxSide ** 2


        # 自己重新实现一遍
        # 相比答案还是比较复杂，以dp[i][j]表示matrix[i][j]作为右下角时最大正方形变成，最后需要判断首行首列等。还是参考答案的解法吧
        # m = len(matrix)
        # n = len(matrix[0])
        # # 初始化dp 首行首列同matrix，dp[i][j]表示matrix[i][j]作为右下角时正方形最大边长
        # dp = [[0] * n for _ in range(m)]
        # dp[0] = [int(c) for c in matrix[0]]
        # for i, row in enumerate(matrix[1:]):
        #     dp[i + 1][0] = int(row[0])
        # # 开始遍历
        # maxSide = float('-inf')
        # for i in range(1, m):
        #     for j in range(1, n):
        #         if matrix[i][j] == '1':
        #             dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        #             maxSide = max(maxSide, dp[i][j])
        # if maxSide == float('-inf'):    # 说明遍历元素都是0
        #     tmp = dp[0] + [row[0] for row in dp[1:]]
        #     return 1 ** 2 if sum(tmp) else 0
        # return maxSide ** 2

        # 自己算是写对了，感觉实现有点复杂
        # m = len(matrix)
        # n = len(matrix[0])
        # dp = [[0] * n for _ in range(m)]
        # dp[0] = [int(c) for c in matrix[0]]
        # for i, row in enumerate(matrix[1:]):
        #     dp[i + 1][0] = int(row[0])
        # print(dp)
        # res = float('-inf')
        # for i in range(1, m):
        #     for j in range(1, n):
        #         if matrix[i][j] == '1':
        #             dp[i][j] = 1 + min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1])
        #             res = max(res, dp[i][j])
        # print(dp)
        # if res == float('-inf'):
        #     tmp = dp[0] + [row[0] for row in dp[1:]]
        #     return 1 if sum(tmp) else 0
        # else:
        #     return res ** 2

    def longestPalindrome(self, s: str) -> str:
        """
        5.最长回文子串
        2023.07.22 中等
        题解：想到 中心探测，先往左边、再往右边、最后同时往两边，就不用考虑奇数个还是偶数个了
        """
        maxSub = ""
        for i in range(len(s)):
            # 往左探
            l = i - 1
            while l >= 0 and s[l] == s[i]:
                l -= 1
            l += 1
            # 往右探
            r = i + 1
            while r <= len(s) - 1 and s[r] == s[i]:
                r += 1
            r -= 1
            # 往两边探
            while l >= 0 and r <= len(s) - 1 and s[l] == s[r]:
                l -= 1
                r += 1
            l += 1
            r -= 1
            maxSub = s[l:r + 1] if len(s[l:r + 1]) > len(maxSub) else maxSub
        return maxSub

    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        """
        139.单词拆分
        2023.07.22 中等
        题解：看了答案，细想一下有动态规划的思想在里面，但这里不是诸位去比较，每次比较是若干个字符，看代码就明白啦
        """
        n = len(s)
        dp = [False for _ in range(n + 1)]      # dp[i]表示前i位能否被字典表示，自然需要到 n
        dp[0] = True        # 前0位肯定能表示，空嘛
        for i in range(n + 1):      # 用来指示dp[i]
            for j in range(i + 1, n + 1):       # 前i位的后若干位是否也能表示(不像之前的动态规划一位一位的去比较，这里用一个for循环)
                if dp[i] and s[i:j] in wordDict:    # dp[i]的i与s[i:j]的i，意义上刚好差1位，你品品
                    dp[j] = True
        return dp[-1]

    def longestPalindromeSubseq(self, s: str) -> int:
        """
        516.最长回文子序列
        2023.07.23 中等
        题解：没做过 注意dp的构造 https://leetcode.cn/problems/longest-palindromic-subsequence/solutions/930442/zui-chang-hui-wen-zi-xu-lie-by-leetcode-hcjqp/?envType=study-plan-v2&envId=dynamic-programming
        """
        # n = len(s)
        # dp = [[0] * n for _ in range(n)]    # dp[i][j]代表s下标[i,j]闭区间范围内最长回文子序列
        # for i in range(n - 1, -1, -1):      # i往左移动，所以从右开始
        #     dp[i][i] = 1    # 每个字符都是回文子序列
        #     for j in range(i + 1, n):
        #         if s[i] == s[j]:
        #             dp[i][j] = dp[i + 1][j - 1] + 2     # 不用担心j跑到i左边去，因为i > j的dp均为0
        #         else:
        #             dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])      # 如果不等，就看看附近刚刚扫描过的多长
        # return dp[0][-1]

        # 重写一遍
        n = len(s)
        dp = [[0] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            dp[i][i] = 1
            for j in range(i + 1, n):
                if s[i] == s[j]:
                    dp[i][j] = dp[i + 1][j - 1] + 2
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
        return dp[0][-1]

    def minimumDeleteSum(self, s1: str, s2: str) -> int:
        """
        712.两个字符串的最小ASCII删除和
        2023.07.23 中等
        题解：同样是注意dp的构造 https://leetcode.cn/problems/minimum-ascii-delete-sum-for-two-strings/solutions/1712998/liang-ge-zi-fu-chuan-de-zui-xiao-asciish-xllf/?envType=study-plan-v2&envId=dynamic-programming
        """
        m = len(s1)
        n = len(s2)
        # 因为dp要表示2个字符串前几个的概念，当然包括 前0个 的概念
        # dp[0][0]、dp[0][j]、dp[i][0]要分别初始化，切记不要出错
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j - 1] + ord(s2[j - 1])
        for i in range(1, m + 1):
            dp[i][0] = dp[i - 1][0] + ord(s1[i - 1])
        # 开始遍历
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # 有1个字符串为空，另一个也只能为空二者才能相等；也算在初始化dp
                if i == 0:
                    dp[i][j] = dp[i][j - 1] + ord(s2[j - 1])     # 不为空的字符要添加一个新字符，那么dp结果是没加之前+添加这个字符的ACSII值
                elif j == 0:
                    dp[i][j] = dp[i - 1][j] + ord(s1[i - 1])
                elif s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                elif s1[i - 1] != s2[j - 1]:
                    dp[i][j] = min(dp[i - 1][j] + ord(s1[i - 1]),   # dp[i - 1][j] s1的前i-1个字符、s2的前j个字符，是没包括s1[i - 1]的
                                   dp[i][j - 1] + ord(s2[j - 1]))   # 同上
        return dp[-1][-1]

    def lengthOfLIS(self, nums: List[int]) -> int:
        """
        300.最长递增子序列
        2023.07.23 中等
        题解：一看之前代码秒懂 自己能想到动态规划，但没想到结果怎么写 注意最内层计算的dp[i]而不是dp[j]
            https://leetcode.cn/problems/longest-increasing-subsequence/solutions/24173/zui-chang-shang-sheng-zi-xu-lie-dong-tai-gui-hua-2/
        """
        n = len(nums)
        dp = [1] * n    # dp[i]表示截止到nums[i]最长子序列的长度
        # 遍历计算dp：对每个i，逐个计算是否能放到前面的dp后面
        for i in range(n):
            for j in range(i):
                if nums[j] < nums[i]:   # nums[i]大，能放到递增子序列后面——即这里计算的是nums[i]，而不是nums[j]！
                    dp[i] = max(dp[i], dp[j] + 1)       # 动态规划是用之前的状态更新现在的状态，这里dp[j]是之前的状态，现在nums[i]又可以放到nums[j]后面，所以+1
        return max(dp)          # 最后返回最大的递增子序列

        # 重写一遍
        # n = len(nums)
        # dp = [1] * n  # 数字不需深拷贝；截止到nums[i]的最长递增子序列；初始化为1，因为每个数也是递增子序列
        # # 遍历计算 dp[i]
        # for i in range(n):
        #     for j in range(i):  # 能否把nums[i]放到每个nums[j]后面
        #         if nums[j] < nums[i]:  # nums[i]能加到后面
        #             dp[i] = max(dp[i], dp[
        #                 j] + 1)  # dp[i]会不断更新，通过nums[i]能否放到每个nums[j]后面，通过dp[j]的长度更新dp[i]；dp[j]就是第一层for计算得到的dp[i]，领会意思
        # return max(dp)

    def findNumberOfLIS(self, nums: List[int]) -> int:
        """
        673.最长递增子序列的个数
        2023.07.23 中等
        题解：与 300.最长递增子序列 很像，看答案 https://leetcode.cn/problems/number-of-longest-increasing-subsequence/solutions/1007075/zui-chang-di-zeng-zi-xu-lie-de-ge-shu-by-w12f/?envType=study-plan-v2&envId=dynamic-programming
        """
        n = len(nums)
        dp = [1] * n        # 截止到nums[i]的最长递增子序列 的长度
        cnt = [1] * n       # 截止到nums[i]的最长递增子序列 的长度 的出现次数
        for i in range(n):
            for j in range(i):
                if nums[j] < nums[i]:   # nums[i]可以放到nums[j]后面
                    if dp[j] + 1 > dp[i]:       # 若大于，则需更新dp[i]，意味着之前dp[i]长度的组合方式丢弃，因为现在有更长的组合方式，所以要为cnt[i]重新赋值而不是累加
                        dp[i] = dp[j] + 1
                        cnt[i] = cnt[j]
                    elif dp[j] + 1 == dp[i]:    # 若等于，不必更新dp[i]，意味着之前dp[i]长度的的组合方式、现在dp[j]+1的组合方式都有效，所以累加cnt[i] + cnt[j]
                        cnt[i] += cnt[j]
        maxLen = max(dp)
        return sum([cnt[i] for i, length in enumerate(dp) if length == maxLen])     # 最大递增子序列长度是maxLen，所有能组合成这个长度的组合方式有多少种

    def findLongestChain(self, pairs: List[List[int]]) -> int:
        """
        646.最长数对链
        2023.07.23 中等
        题解：没做出来可以先看下答案，并不难 https://leetcode.cn/problems/maximum-length-of-pair-chain/solutions/1793617/zui-chang-shu-dui-lian-by-leetcode-solut-ifpn/?envType=study-plan-v2&envId=dynamic-programming
        """
        pairs = sorted(pairs, key=lambda x: x[0])
        n = len(pairs)
        dp = [1] * n
        for i in range(n):
            for j in range(i):
                if pairs[j][-1] < pairs[i][0]:      # pairs[i]可以放到后面
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)

    def longestSubsequence(self, arr: List[int], difference: int) -> int:
        """
        1218.最长定差子序列
        2023.07.24 中等
        题解：或直接看答案 尝试一下
        """
        from collections import defaultdict
        # 看答案，代码特别简单
        # 以元素值为key
        dp = defaultdict(int)   # 工厂函数，可直接获取任意key而不担心报错，value默认为0
        # 因为要计算等差子序列，我们总需要获取当前遍历v的 前一个等差值v-difference 的dp，而不是将每个arr[i]放到arr[j]后去比较(像 300.)，这样也减少了复杂度
        for v in arr:
            dp[v] = dp[v - difference] + 1
        return max(dp.values())

        # 思路同 300.等题 直接使用会超时
        # n = len(arr)
        # dp = [1] * n
        # for i in range(n):
        #     for j in range(i):
        #         if arr[i] - arr[j] == difference:
        #             dp[i] = max(dp[i], dp[j] + 1)
        # return max(dp)

    def longestArithSeqLength(self, nums: List[int]) -> int:
        """
        1027.最长等差数列
        2023.07.24 中等
        题解：https://leetcode.cn/problems/longest-arithmetic-subsequence/?envType=study-plan-v2&envId=dynamic-programming
        """
        # 代码从内层for开始看
        minV, maxV = min(nums), max(nums)
        difference = maxV - minV
        res = 0
        for d in range(-difference, difference + 1):
            # 固定差值d后，进行动态规划遍历
            f = dict()
            for n in nums:
                # 当前遍历值n的前一个数n-d是否存在f中，则有f(n)=f(n-d)+1，不存在则f(n)=1
                if n - d in f:
                    f[n] = max(0, f[n - d] + 1)
                    res = max(res, f[n])
                f[n] = max(f.get(n, 0), 1)      # 若当前遍历的值不存在于f中，置1
        return res

    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        """
        1143.最长公共子序列
        2023.07.24 中等
        题解：二维动态规划 https://leetcode.cn/problems/longest-common-subsequence/solutions/696763/zui-chang-gong-gong-zi-xu-lie-by-leetcod-y7u0/?envType=study-plan-v2&envId=dynamic-programming
        """
        m = len(text1)
        n = len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[-1][-1]

    def maxUncrossedLines(self, nums1: List[int], nums2: List[int]) -> int:
        """
        1035.不相交的线
        2023.07.24 中等
        题解：能想到与上题 1143.最长公共子序列 解题思路可能相同--事实证明，一模一样
        """
        m = len(nums1)
        n = len(nums2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]      # nums1的前i个字符、nums2的前j个字符
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if nums1[i - 1] == nums2[j - 1]:        # 相等，就可以放到二者公共子序列后面
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:       # 不等，指定不能同时放了，则用 放i指向的元素、不放j指向的元素即dp[i][j-1] 和 放j指向的元素、不放i指向的元素即dp[i-1][j] 的较大者更新dp[i][j]
                    dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])      # 这里面包含着 都不放 的意思，若写成都不放就错了，有时候只放一个公共子序列可能会更长
        return dp[-1][-1]

    def maxProfit(self, prices: List[int]) -> int:
        """
        309.最佳买卖股票时机含冷冻期
        2023.07.24 中等
        题解：再看一遍官方解析吧；我没看过答案的代码，牛逼！
            依旧是每天三个状态，官方答案的解析比以前更清楚 https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-with-cooldown/solutions/323509/zui-jia-mai-mai-gu-piao-shi-ji-han-leng-dong-qi-4/?envType=study-plan-v2&envId=dynamic-programming
        """
        # 每天的三种状态：
        #   1.持有
        #   2.不持有，处于冷冻期
        #   3.不持有，不处于冷冻期
        # dp = [[0, 0, 0] for _ in range(len(prices))]
        # dp[0] = [-prices[0], 0, 0]      # 初始化第一天：若持有只能是第一天买入的嘛；不持有也不用区分是否在冷冻期，因为第一天嘛
        # for i in range(1, len(prices)):
        #     dp[i][0] = max(dp[i - 1][0], dp[i - 1][2] - prices[i])
        #     dp[i][1] = dp[i - 1][0] + prices[i]     # 今天不持有且处于冷冻期，只能是昨天持有今天卖了
        #     dp[i][2] = max(dp[i - 1][1], dp[i - 1][2])      # 今天不持有且不处于冷冻期，只能是昨天就不持有
        # return max(dp[-1][1:])

        # 重复一遍
        # dp = [[0, 0, 0] for _ in range(len(prices))]
        # dp[0] = [-prices[0], 0, 0]
        # for i in range(1, len(prices)):
        #     dp[i][0] = max(dp[i - 1][0], dp[i - 1][2] - prices[i])
        #     dp[i][1] = dp[i - 1][0] + prices[i]
        #     dp[i][2] = max(dp[i - 1][1], dp[i - 1][2])
        # return max(dp[-1][1:])

        # 再重复一遍，不嫌烦！
        n = len(prices)
        # 3种状态：1.持有 2.不持有，处于冷冻期 3.不持有，不在冷冻期
        dp = [[0, 0, 0] for _ in range(n)]
        dp[0] = [-prices[0], 0, 0]
        for i in range(1, n):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][2] - prices[i])
            dp[i][1] = dp[i - 1][0] + prices[i]
            dp[i][2] = max(dp[i - 1][2], dp[i - 1][1])
        return max(dp[-1][1:])

    def maxProfit(self, prices: List[int], fee: int) -> int:
        """
        714.买卖股票的最佳时机含手续费
        2023.07.25 中等
        题解：状态找不对，本题只有2个状态 https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/solutions/524669/mai-mai-gu-piao-de-zui-jia-shi-ji-han-sh-rzlz/?envType=study-plan-v2&envId=dynamic-programming
        """
        n = len(prices)
        # 本题只2种状态：1.不持有 2.持有
        dp = [[0, 0] for _ in range(n)]
        dp[0] = [0, -prices[0] - fee]
        for i in range(1, n):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i] - fee)
        return dp[-1][0]

    def numTrees(self, n: int) -> int:
        """
        96.不同的二叉搜索树
        2023.07.25 中等
        题解：想不起来思路，一看代码就懂了
        """
        dp = [0] * (n + 1)    # dp[i]代表i个节点能组成多少不同的BST
        dp[0] = 1
        dp[1] = 1
        for i in range(2, n + 1):       # 遍历计算dp[i]
            for j in range(i):      # 其中一棵子树使用j个节点，则另一棵子树使用i-1-j个节点(根节点还占一个呢)
                dp[i] += dp[j] * dp[i - 1 - j]       # 这里是 += 而不是 =，要把i个节点能组成几种不同BST的情况都考虑一遍，这些情况的结果是要累加的
        return dp[-1]

    def generateTrees(self, n: int) -> List[Optional[TreeNode]]:
        """
        95.不同的二叉搜索树II
        2023.07.25 中等
        题解：
        """
        def func(start, end):
            if start > end:
                return [None, ]
            res = []
            for i in range(start, end + 1):
                lefts = func(start, i - 1)
                rights = func(i + 1, end)
                for l in lefts:
                    for r in rights:
                        root = TreeNode(i)
                        root.left = l
                        root.right = r
                        res.append(root)
            return res

        res = func(1, n)
        return res

    def rob(self, root: Optional[TreeNode]) -> int:
        """
        337.打家劫舍II
        2023.07.25 中等
        题解：递归回溯，是动态规划吗
        """
        def func(node):
            if not node:
                return [0, 0]
            # 注意leftLs、rightLs返回的形式
            leftLs = func(node.left)
            rightLs = func(node.right)
            return [max(leftLs) + max(rightLs),         # 不偷当前节点
                    node.val + leftLs[0] + rightLs[0]]  # 偷当前节点

        return max(max(func(root.left)) + max(func(root.right)),            # 不偷根节点
                   root.val + func(root.left)[0] + func(root.right)[0])     # 偷根节点

        # 找一下动态规划的解法


if __name__ == "__main__":
    sl = Solution()

    print(sl.numTrees(1))
