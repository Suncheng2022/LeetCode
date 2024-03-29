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
        https://leetcode.cn/problems/maximal-square/solutions/44586/li-jie-san-zhe-qu-zui-xiao-1-by-lzhlyle/
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
        题解：https://leetcode.cn/problems/longest-arithmetic-subsequence/solutions/2238031/zui-chang-deng-chai-shu-lie-by-leetcode-eieq8/?envType=study-plan-v2&envId=dynamic-programming
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
            https://leetcode.cn/problems/unique-binary-search-trees/solutions/330990/shou-hua-tu-jie-san-chong-xie-fa-dp-di-gui-ji-yi-h/?envType=featured-list&envId=2cktkvj        """
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
        题解：https://leetcode.cn/problems/unique-binary-search-trees-ii/solutions/339143/bu-tong-de-er-cha-sou-suo-shu-ii-by-leetcode-solut/?envType=study-plan-v2&envId=dynamic-programming
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
        337.打家劫舍III
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

    def numSquares(self, n: int) -> int:
        """
        279.完全平方数【超时】
        2023.07.25 中等
        题解：n由多少个完全平方数相加，n-1*1由多少个完全平方数相加，n-2*2、n-3*3...
        """
        # 官方答案C++
        dp = [0] * (n + 1)
        for i in range(1, n + 1):
            minn = float('inf')
            j = 1
            while i - j ** 2 >= 0:
                minn = min(minn, dp[i - j ** 2])
                j += 1
            dp[i] = minn + 1    # minn得到的是i-j^2所需最小完全平方数的和
        return dp[-1]

        # 超时 应该不是代码的原因
        # dp = [0] * (n + 1)      # dp[0]肯定是0啦；dp[i]  i最少可由多少个完全平方数相加组成
        # for i in range(1, n + 1):
        #     dp[i] = i       # 先假设有i个1x1相加，这是最多的可能
        #     j = 1
        #     while i - j ** 2 >= 0:      # 通过尝试不断去更新缩小dp[i]
        #         dp[i] = min(dp[i], dp[i - j ** 2] + 1)
        #         j += 1
        # return dp[n]

    def change(self, amount: int, coins: List[int]) -> int:
        """
        518.零钱兑换
        2023.07.25 中等
        题解：遍历硬币、遍历更新dp[i] https://leetcode.cn/problems/coin-change-ii/solutions/821278/ling-qian-dui-huan-ii-by-leetcode-soluti-f7uh/?envType=study-plan-v2&envId=dynamic-programming
        """
        # 再写一遍
        dp = [0] * (amount + 1)
        dp[0] = 1
        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] += dp[i - coin]       # 本题求几种硬币组合，而不是几个硬币，所以这里不+1
        return dp[-1]

        # 重写一遍
        # dp = [0] * (amount + 1)
        # dp[0] = 1       # 【注意】细节，金额为0的使用0个硬币，只有这一种组合
        # for coin in coins:      # 遍历硬币
        #     for i in range(coin, amount + 1):   # 考虑所有 金额>=coin，计算所有能把coin放进去的方式
        #         dp[i] += dp[i - coin]       # 所有金额为i-coin的，加上一个coin硬币就能组成金额i
        # return dp[-1]

        # dp = [0] * (amount + 1)
        # dp[0] = 1       # 这是个细节，很重要，自己想不到
        # for coin in coins:
        #     for i in range(coin, amount + 1):       # i指示更新哪个dp；包含了
        #         dp[i] += dp[i - coin]       # i-coin表示当前组合加入coin硬币后正好满足金额i，对所有能考虑的i，就是范围coin<=i<=amount；如果组成金额i时，前面有金额i-coin的结果，用来更新即可
        # return dp[-1]

    def combinationSum4(self, nums: List[int], target: int) -> int:
        """
        377.组合总和IV
        2023.07.26 中等
        题解：相比 518.零钱兑换，本题将组合的不同顺序视为不同的结果
            看答案 https://leetcode.cn/problems/combination-sum-iv/solutions/740581/zu-he-zong-he-iv-by-leetcode-solution-q8zv/?envType=study-plan-v2&envId=dynamic-programming
        """
        dp = [0] * (target + 1)
        dp[0] = 1
        # 先遍历组合的和，再考虑后面是否能添加nums的一个数，这样就考虑了组合顺序
        for i in range(1, target + 1):
            for n in nums:
                if n <= i:
                    dp[i] += dp[i - n]      # i-n后面放上n能组成i
        return dp[-1]

    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        """
        474.一和零
        2023.07.26 中等
        题解：背包问题 三维背包问题 三维动态规划 https://leetcode.cn/problems/ones-and-zeroes/solutions/814806/yi-he-ling-by-leetcode-solution-u2z2/?envType=study-plan-v2&envId=dynamic-programming
        """
        from collections import Counter

        dp = [[[0] * (n + 1) for _ in range(m + 1)] for _ in range(len(strs) + 1)]
        for i in range(1, len(strs) + 1):
            zeros = Counter(strs[i - 1])['0']       # 这里是strs[i-1]
            ones = Counter(strs[i - 1])['1']
            for j in range(m + 1):                  # 这里是从几个0开始迭代更新
                for k in range(n + 1):              # 这里是从几个1开始迭代更新
                    if j < zeros or k < ones:
                        dp[i][j][k] = dp[i - 1][j][k]   # 当前字符串中'0' '1'的数量超过了j k，所以不能选
                    elif j >= zeros and k >= ones:
                        dp[i][j][k] = max(dp[i - 1][j][k], dp[i - 1][j - zeros][k - ones] + 1)      # 当前字符串strs[i-1]可选可不选，取结果较大的结果
        return dp[-1][-1][-1]

    def mostPoints(self, questions: List[List[int]]) -> int:
        """
        2140.解决智力问题
        2023.07.26 中等
        题解：答案 提示2 反向动态规划 https://leetcode.cn/problems/solving-questions-with-brainpower/solutions/1233147/jie-jue-zhi-li-wen-ti-by-leetcode-soluti-ieuq/?envType=study-plan-v2&envId=dynamic-programming
        """
        # n = len(questions)
        # dp = [0] * (n + 1)      # dp[n]=0作为边界条件初始化；dp[i]表示解决i及以后的题目能得到的最高分数；索引i 同时也是 题目索引
        # for i in range(n - 1, -1, -1):   # 这里索引范围思考一下
        #     dp[i] = max(questions[i][0] + dp[min(n, i + questions[i][1] + 1)],      # 当前题 做，则后面的能做的是i+ques[i][1]+1 对的
        #                 dp[i + 1])      # 当前题 不做，直接使用上一个做了的题的结果
        # return dp[0]

        # 再写一遍
        n = len(questions)
        dp = [0] * (n + 1)      # dp[n]=0，已经不包含题目了
        for i in range(n - 1, -1, -1):
            dp[i] = max(questions[i][0] + dp[min(n, i + questions[i][1] + 1)], dp[i + 1])
        return dp[0]

    def coinChange(self, coins: List[int], amount: int) -> int:
        """
        322.零钱兑换
        2023.07.26 中等
        题解：答案 方法二 动态规划 dp[i]：金额i所需最少的硬币数
            https://leetcode.cn/problems/coin-change/solutions/132979/322-ling-qian-dui-huan-by-leetcode-solution/
        """
        # 动态规划
        dp = [float('inf')] * (amount + 1)     # 用一个较大的数初始化dp；dp[i] 组成金额i需要最少硬币数量
        dp[0] = 0
        for i in range(1, amount + 1):
            for coin in coins:
                if i - coin >= 0:
                    dp[i] = min(dp[i], dp[i - coin] + 1)
        return dp[-1] if dp[-1] != float('inf') else -1

        # 递归解法 不好理解，不好背
        # def func(amount):
        #     if amount in memo:
        #         return memo[amount]
        #     elif amount == 0:
        #         return 0
        #     elif amount < 0:
        #         return -1
        #
        #     best = float('inf')     # 用于记录这一轮遍历的最优值
        #     for coin in coins:
        #         subproblem = func(amount - coin)    # subproblem少了一个面值为coin的硬币
        #         if subproblem < 0:      # 小于0说明无解，换硬币继续尝试
        #             continue
        #         best = min(best, subproblem + 1)    # subproblem不是少一个硬币的计算结果，再加上就是了
        #     memo[amount] = best if best != float('inf') else -1
        #     return memo[amount]
        #
        # memo = {}  # 备忘录，记录计算过值，减少计算复杂度
        # return func(amount)

    def countGoodStrings(self, low: int, high: int, zero: int, one: int) -> int:
        """
        2466.统计构造好字符串的方案数
        2023.07.26 中等
        题解：乍一看比较难 题解说类似 70.爬楼梯
            爬楼梯一次可以爬1阶或2阶，这里一次添加zero个'0'字符或one个'1'字符，思路果然一样！
            https://leetcode.cn/problems/count-ways-to-build-good-strings/solutions/1964910/by-endlesscheng-4j22/?envType=study-plan-v2&envId=dynamic-programming
        """
        MOD = 10 ** 9 + 7
        dp = [0] * (high + 1)
        dp[0] = 1
        for i in range(1, high + 1):
            if i >= zero:
                dp[i] += dp[i - zero]
            if i >= one:
                dp[i] += dp[i - one]
        return sum(dp[low:]) % MOD

    def numDecodings(self, s: str) -> int:
        """
        91.解码方法
        2023.07.26 中等
        题解：考虑上次解码是1位or2位 https://leetcode.cn/problems/decode-ways/solutions/734344/jie-ma-fang-fa-by-leetcode-solution-p8np/?envType=study-plan-v2&envId=dynamic-programming
        """
        dp = [0] * (len(s) + 1)     # 解码前i位的方法数量
        dp[0] = 1
        for i in range(1, len(s) + 1):      # 遍历计算dp[i]；当前处理的是s[i]
            # 解码可选择 解码1位 和 解码2位；和--意味着两种方式要同时考虑！
            if s[i - 1] != '0':
                dp[i] += dp[i - 1]                                      # 解码1位
            if i > 1 and s[i - 2] != '0' and int(s[i - 2:i]) <= 26:     # 解码2位
                dp[i] += dp[i - 2]
        return dp[-1]

    def mincostTickets(self, days: List[int], costs: List[int]) -> int:
        """
        983.最低票价
        2023.07.27 中等
        题解：https://leetcode.cn/problems/minimum-cost-for-tickets/solutions/233810/zui-di-piao-jie-by-leetcode-solution/?envType=study-plan-v2&envId=dynamic-programming
        """
        # 答案 递归
        days = set(days)
        memo = {}

        def func(i):
            """ 计算dp[i] 从第i天开始到今年结束所需花费 """
            if i in memo:
                return memo[i]

            if i > 365:
                return 0
            elif i in days:
                res = min(func(i + d) + c for d, c in zip([1, 7, 30], costs))
                memo[i] = res
                return memo[i]
            else:
                return func(i + 1)

        return func(1)

        # 没能全过
        # dp = [0] * 366      # 1~365天；dp[i]表示从第i天开始到今年结束所需花费
        # days = set(days)
        # for i in range(365, 0, -1):     # 反序动态规划
        #     if i in days:       # 如果这天需要旅行，就得买票啦。怎么买花费最小，细细品味
        #         dp[i] = min(dp[i + d] + c for d, c in zip([1, 7, 30], costs) if i + d <= 365) if i + 1 <= 365 else costs[0]
        #     else:               # 如果这天不需要旅行，继承dp[i+1]即可
        #         dp[i] = dp[i + 1] if i + 1 <= 365 else 0
        # return dp[1]        # 最后返回 从第1天开始到今年结束 所需花费

    def numTilings(self, n: int) -> int:
        """
        790.多米诺和托米诺平铺
        2023.07.27 中等
        题解：很难 https://leetcode.cn/problems/domino-and-tromino-tiling/solutions/1962465/duo-mi-nuo-he-tuo-mi-nuo-ping-pu-by-leet-7n0j/?envType=study-plan-v2&envId=dynamic-programming
            dp[0]的初始化：i从1开始计数，就是0时表示一列都没有，相当于填满
        """
        MOD = 10 ** 9 + 7
        dp = [[0] * 4 for _ in range(n + 1)]
        dp[0][3] = 1
        for i in range(1, n + 1):
            dp[i][0] = dp[i - 1][3] % MOD
            dp[i][1] = (dp[i - 1][0] + dp[i - 1][2]) % MOD
            dp[i][2] = (dp[i - 1][0] + dp[i - 1][1]) % MOD
            dp[i][3] = (dp[i - 1][3] + dp[i - 1][0] + dp[i - 1][1] + dp[i - 1][2]) % MOD
        return dp[-1][-1] % MOD


if __name__ == "__main__":
    sl = Solution()

    days = [364]
    costs = [3, 3, 1]
    print(sl.mincostTickets(days, costs))
