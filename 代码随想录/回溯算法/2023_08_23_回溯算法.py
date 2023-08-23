from typing import List

class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        """ 47.全排列II
            排列问题 不使用startInd 因为要考虑不同元素顺序
                    去重 排序、used """
        def backtracking():
            if len(path) == len(nums):
                res.append(path[:])
                return
            for i in range(len(nums)):
                if i > 0 and nums[i - 1] == nums[i] and not used[i - 1]:
                    continue
                if not used[i]:
                    path.append(nums[i])
                    used[i] = True
                    backtracking()
                    path.pop()
                    used[i] = False

        res = []
        path = []
        nums.sort()
        used = [False] * len(nums)
        backtracking()
        return res

    def permute(self, nums: List[int]) -> List[List[int]]:
        """ 46.全排列
            排列问题 不使用startInd
                    题目说明nums无重复元素，似乎不用去重
                    used 通过调试运行也可发现 """
        def backtracking():
            if len(path) == len(nums):
                res.append(path[:])
                return
            for i in range(len(nums)):
                if used[i]:
                    continue
                path.append(nums[i])
                used[i] = True
                backtracking()
                path.pop()
                used[i] = False

        res = []
        path = []
        used = [False] * len(nums)
        backtracking()
        return res

    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        """ 491.递增子序列
            子序列  不能重复选 {2,1} {1,2}视为相同，因此需要startIndex
                   去重 通过调试运行发现的[但此题万万不可排序] """
        def backtracking(startInd):
            if len(path) > 1:
                res.append(path[:])
            if startInd >= len(nums):
                return

            for i in range(startInd, len(nums)):
                if (path and path[-1] > nums[i]) or nums[i] in used:
                    continue
                path.append(nums[i])
                used.append(nums[i])
                backtracking(i + 1)
                used.pop()
                path.pop()

        res = []
        path = []
        used = []
        backtracking(0)
        return res


if __name__ == '__main__':
    sl = Solution()

    nums = [4,6,7,7]
    print(sl.findSubsequences(nums))