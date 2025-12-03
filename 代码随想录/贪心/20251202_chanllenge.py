"""
怕吗?
"""
from typing import List


class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        """ 643.子数组最大平均数I """
        ## 想到滑窗
        if len(nums) == 1:
            return nums[0]
        curAvg = maxAvg = sum(nums[:k])
        for i in range(k, len(nums)):
            curAvg = curAvg - nums[i - k] + nums[i]
            maxAvg = max(maxAvg, curAvg)
        return maxAvg / k
    
    def lengthOfLIS(self, nums: List[int]) -> int:
        """ 300.最长递增子序列 """
        n = len(nums)
        dp = [1] * n        # dp[i] 以nums[i]为结尾的最长递增子序列长度
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)
    
    def findLengthOfLCIS(self, nums: List[int]) -> int:
        """ 674.最长连续递增序列 \n
            300.最长递增子序列 不要求连续, 本题要求连续 """
        n = len(nums)
        dp = [1] * n        # dp[i] 以nums[i]结尾的最长连续递增子序列的长度
        for i in range(1, n):
            if nums[i] > nums[i - 1]:
                dp[i] = dp[i - 1] + 1
        return max(dp)

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        """ 15.三数之和 """
        nums.sort()
        res = []
        for i in range(len(nums) - 2):
            if i > 0 and nums[i - 1] == nums[i]:
                continue
            l, r = i + 1, len(nums) - 1
            while l < r:
                _tmp = [nums[i], nums[l], nums[r]]
                _sum = sum(_tmp)
                if _sum == 0:
                    res.append(_tmp[:])
                    while l < r and nums[l + 1] == nums[l]:
                        l += 1
                    while l < r and nums[r - 1] == nums[r]:
                        r -= 1
                    l += 1
                    r -= 1
                elif _sum < 0:
                    l += 1
                else:
                    r -= 1
        return res
    
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        """ 16.最接近的三数之和 \n
            与 15.三数之和 基本相同 """
        nums.sort()
        res = sum(nums[:3])
        for i in range(len(nums) - 2):
            if i > 0 and nums[i - 1] == nums[i]:
                continue
            l, r = i + 1, len(nums) - 1
            while l < r:
                _sum = nums[i] + nums[l] + nums[r]
                res = _sum if abs(target - _sum) < abs(target - res) else res
                if _sum == target:
                    return res
                elif _sum < target:
                    l += 1
                else:
                    r -= 1
        return res

if __name__ == '__main__':
    sl = Solution()

    nums = [4,0,4,3,3]
    k = 5
    print(sl.findMaxAverage(nums, k))