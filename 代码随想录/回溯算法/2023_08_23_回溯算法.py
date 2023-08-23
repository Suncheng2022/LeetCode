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
                1.很明显不能重复使用元素，所以使用startIndex控制下层递归元素的起始索引
                2.同一层使用过的元素不能重复使用，注意 是 同一层, 因此要确定used的位置 """
        def backtracking(startInd):
            if len(path) > 1:
                res.append(path[:])
            if startInd >= len(nums):
                return
            used = set()        # 本层元素不能重复用 否则会产生相同结果
            for i in range(startInd, len(nums)):
                if (path and path[-1] > nums[i]) or nums[i] in used:
                    continue
                used.add(nums[i])
                path.append(nums[i])
                backtracking(i + 1)
                path.pop()

        res = []
        path = []
        backtracking(0)
        return res

    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        """ 90.子集II """
        def backtracking(startIndex):
            res.append(path[:])
            if startIndex >= len(nums):
                return
            for i in range(startIndex, len(nums)):
                if i > 0 and nums[i - 1] == nums[i] and not used[i - 1]:    # 同层使用过相同元素，跳过
                    continue
                used[i] = True
                path.append(nums[i])
                backtracking(i + 1)
                path.pop()
                used[i] = False

        res = []
        path = []
        used = [False] * len(nums)
        nums.sort()
        backtracking(0)
        return res

    def subsets(self, nums: List[int]) -> List[List[int]]:
        """ 78.子集 """
        def backtracking(startIndex):
            res.append(path[:])
            if startIndex >= len(nums):
                return
            for i in range(startIndex, len(nums)):
                path.append(nums[i])
                backtracking(i + 1)
                path.pop()

        res = []
        path = []
        backtracking(0)
        return res

    def restoreIpAddresses(self, s: str) -> List[str]:
        """ 93.复原IP地址 """
        # def backtracking(startIndex):
        #     if len(path) == 4:
        #         res.append(".".join(path))
        #         return
        #     for i in range(startIndex, len(nums)):
        #
        #
        # res = []
        # path = []



if __name__ == '__main__':
    sl = Solution()

    nums = [1]
    print(sl.subsets(nums))