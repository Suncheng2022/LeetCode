'''
10.26   执行力这么差?
'''
from typing import List

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        """ 1.两数之和 """
        for i in range(len(nums)):
            if target - nums[i] in nums[i + 1:]:
                return [i, nums.index(target - nums[i], i + 1)]
    
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        """ 49.字母异位词分组 """
        from collections import defaultdict

        mydict = defaultdict(list)
        for word in strs:
            k = ''.join(sorted(word))
            mydict[k].append(word)
        return list(mydict.values())
    
    def longestConsecutive(self, nums: List[int]) -> int:
        """ 128.最长连续序列 """
        nums = set(nums)
        res = 0
        for dg in nums:
            if dg - 1 not in nums:      # 保证dg是序列开头
                _len = 1
                while dg + 1 in nums:
                    _len += 1
                    dg += 1
                res = max(res, _len)
        return res

    def moveZeroes(self, nums: List[int]) -> None:
        """
        283.移动零 \n
        Do not return anything, modify nums in-place instead.
        """
        ## 指定放置非零元素的位置.
        ## 不会发生覆盖问题
        ind = 0
        for i in range(len(nums)):      # i比ind要快
            if nums[i] != 0:
                nums[ind] = nums[i]
                ind += 1
        for i in range(ind, len(nums)):
            nums[i] = 0

        ## 将0后移 或 0前移 的想法, 走不通, 解决不了相邻元素为零/非零的情况

    def maxArea(self, height: List[int]) -> int:
        """ 11.盛最多水的容器 """
        l, r = 0, len(height) - 1
        maxArea = (r - l) * min(height[l], height[r])
        while l < r:
            if height[l] < height[r]:       # 要移动矮的柱子, 这样才有机会容量更大. (若移动较高的柱子, 则最矮的柱子不会变, 同时宽度在减小, 所以不可能出现更大容量)
                l += 1
            else:
                r -= 1
            curArea = (r - l) * min(height[l], height[r])
            maxArea = max(maxArea, curArea)
        return maxArea
    
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        """ 15.三数之和 """
        nums.sort()
        res = []
        for i in range(len(nums) - 2):
            if i > 0 and nums[i] == nums[i - 1]:                # 解超时
                continue
            l, r = i + 1, len(nums) - 1
            while l < r:
                _tmp = [nums[i], nums[l], nums[r]]
                _sum = sum(_tmp)
                if _sum == 0:
                    res.append(_tmp)                            # 前面跳过重复计算了, 那就不会有重复结果了, 可以直接加入
                    while l < r and nums[l] == nums[l + 1]:     # 解超时
                        l += 1
                    while l < r and nums[r] == nums[r - 1]:     # 解超时
                        r -= 1
                    l += 1
                    r -= 1
                elif _sum < 0:
                    l += 1
                else:
                    r -= 1
        return res
    
    def lengthOfLongestSubstring(self, s: str) -> int:
        """ 3.无重复字符的最长子串 \n
            要求连续 """
        ## 高效 + 边界更容易考虑
        # maxLength = 0
        # for i in range(len(s)):
        #     seen = set()
        #     for j in range(i, len(s)):
        #         if s[j] in seen:
        #             break
        #         seen.add(s[j])
        #     maxLength = max(maxLength, len(seen))
        # return maxLength

        ## 低效, 且考虑边界的方式复杂
        # if len(s) <= 1:
        #     return len(s)
        # maxLength = 0
        # for i in range(len(s)):
        #     _cur = s[i]
        #     for j in range(i + 1, len(s)):
        #         if s[j] in _cur:
        #             maxLength = max(maxLength, len(_cur))
        #             break
        #         else:
        #             _cur += s[j]
        #             maxLength = max(maxLength, len(_cur))
        # return maxLength

        ## Again, 重复上面高效的实现
        maxLength = 0
        for i in range(len(s)):
            seen = set()            # 滑动窗口 + set查找高效
            for j in range(i, len(s)):
                if s[j] in seen:
                    break
                seen.add(s[j])
            maxLength = max(maxLength, len(seen))
        return maxLength
        
    def findAnagrams(self, s: str, p: str) -> List[int]:
        """ 438.找到字符串中所有字母异位词 """
        ## 参考之前的实现
        res = []
        n = len(s)
        m = len(p)
        if n < m:
            return res
        
        # 初始化滑窗
        s_count = [0] * 26
        p_count = [0] * 26
        for i in range(m):      
            s_count[ord(s[i]) - ord('a')] += 1
            p_count[ord(p[i]) - ord('a')] += 1
        
        if s_count == p_count:
            res.append(0)
        for i in range(m, n):
            s_count[ord(s[i]) - ord('a')] += 1      # 进滑窗
            s_count[ord(s[i - m]) - ord('a')] -= 1  # 出滑窗
            if s_count == p_count:
                res.append(i - m + 1)               # 出滑窗元素的索引是i-m, 则当前滑窗起始索引是i-m+1
        return res

        ## 能过, 但效率太低
        # res = []
        # m = len(p)
        # _p = ''.join(sorted(p))
        # for i in range(0, len(s) - m + 1):
        #     _s = ''.join(sorted(s[i:i + m]))
        #     if _s == _p:
        #         res.append(i)
        # return res

    def subarraySum(self, nums: List[int], k: int) -> int:
        """ 560.和为K的子数组 \n
            要求连续 """
        ## 优化效率--前缀和, 记忆累加结果, 计算结果重复利用
        from collections import defaultdict

        preSums = defaultdict(int)      # 前缀和: 前缀和出现的次数
        preSums[0] = 1                  # 初始化, 也很重要
        presum = 0                      # 当前的前缀和
        res = 0                         # 最终结果
        
        for i in range(len(nums)):
            presum += nums[i]
            res += preSums[presum - k]
            preSums[presum] += 1
        return res

        ## 超时
        # res = 0
        # for i in range(len(nums)):
        #     for j in range(i, len(nums)):
        #         if sum(nums[i:j + 1]) == k:
        #             res += 1
        # return res

    def maxSubArray(self, nums: List[int]) -> int:
        """ 53.最大子数组和 \n
            要求连续 """
        ## 动态规划
        dp = [0] * len(nums)        # dp[i] 以nums[i]结尾的连续子数组和为dp[i]
        dp[0] = nums[0]
        for i in range(1, len(nums)):
            dp[i] = max(nums[i], dp[i - 1] + nums[i])
        return max(dp)
    
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        """ 56.合并区间 """
        ## 其实很简单哈
        intervals.sort(key=lambda x: x[0])
        res = [intervals[0]]
        for inter in intervals[1:]:
            if res[-1][-1] >= inter[0]:
                res[-1][-1] = max(res[-1][-1], inter[1])
            else:
                res.append(inter)
        return res
    
    def rotate(self, nums: List[int], k: int) -> None:
        """
        189.轮转数组 \n
        Do not return anything, modify nums in-place instead.
        """
        # nums = A + B
        # 1.整个序列反转 rev(nums) = rev(B) + rev(A)
        # 2.单独对B反转 得 rev(rev(B)) + rev(A)
        # 3.单独对A反转 得 rev(rev(B)) + rev(rev(A)) = B + A, 即为所求
        def rev(i, j):
            while i < j:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
                j -= 1
        
        n = len(nums)
        k %= n
        rev(0, n - 1)
        rev(0, k - 1)
        rev(k, n - 1)

    def productExceptSelf(self, nums: List[int]) -> List[int]:
        """ 238.除自身以外数组的乘积 """
        # 正着乘一遍, 反着乘一遍
        n = len(nums)
        res = [0] * n       # 保存结果
        tmp = 1
        for i in range(n):
            res[i] = tmp    # 先赋值
            tmp *= nums[i]  # 再累乘
        tmp = 1
        for i in range(n - 1, -1, -1):
            res[i] *= tmp
            tmp *= nums[i]
        return res
        

if __name__ == '__main__':
    sl = Solution()

    nums = [1,2,3,4]
    print(sl.productExceptSelf(nums))