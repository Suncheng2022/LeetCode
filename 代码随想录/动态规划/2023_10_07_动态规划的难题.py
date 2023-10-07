from typing import List


class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        """ 300.最长递增子序列
            中等
            dp[i] 以nums[i]结尾的元素 能构成的最长递增子序列的长度
            遍历核心思想：nums[i]能不能放到nums[j]后面 """
        n = len(nums)
        dp = [1] * n
        for i in range(n):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)

    def findLengthOfLCIS(self, nums: List[int]) -> int:
        """ 674.最长连续递增序列
            简单
            dp[i] 以nums[i]结尾能构成的最长连续递增序列的长度
            遍历：只比较相邻元素 """
        n = len(nums)
        dp = [1] * n
        for i in range(1, n):
            if nums[i - 1] < nums[i]:
                dp[i] = dp[i - 1] + 1
        return max(dp)

    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        """ 718.最长重复子数组
            中等
            求 最长'连续'公共子数组 的长度
            dp[i][j] 以索引i-1元素结尾、索引j-1元素结尾的 最长'连续'公共子数组的长度
            '连续'只考虑相等元素的情况即可 """
        m, n = len(nums1), len(nums2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if nums1[i - 1] == nums2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                # else:
                #     dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])
        return max(max(ls) for ls in dp)

    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        """ 1143.最长公共子序列
            中等
            求'不连续'的最长公共子序列的长度
            dp[i][j] 以索引i-1元素结尾、索引j-1元素结尾的 最长'不连续'公共子序列长度
            遍历：'不连续'要考虑元素不等时的情况了 """
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return max(max(ls) for ls in dp)


if __name__ == "__main__":
    sl = Solution()

    text1 = "c"
    text2 = "f"
    print(sl.longestCommonSubsequence(text1, text2))
