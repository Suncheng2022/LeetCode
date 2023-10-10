from typing import List


class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        """ 78.子集
            中等
            子集问题，收集所有节点 """
        path = []
        res = []

        def backtracking(startInd):
            res.append(path[:])
            for i in range(startInd, len(nums)):
                path.append(nums[i])
                backtracking(i + 1)
                path.pop()

        backtracking(0)
        return res

    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        """ 90.子集II
            中等
            nums有重复元素，其余同上 """
        path = []
        res = []
        used = [False] * len(nums)

        def backtracking(startInd, used):
            res.append(path[:])
            for i in range(startInd, len(nums)):
                if i > 0 and nums[i - 1] == nums[i] and not used[i - 1]:
                    continue
                used[i] = True
                path.append(nums[i])
                backtracking(i + 1, used)
                path.pop()
                used[i] = False

        nums.sort()
        backtracking(0, used)
        return res

    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        """ 491.递增子序列
            中等
            nums包含重复元素
                元素不能重复使用--startInd指示递归函数起始位置
                【很重要！】相当于遍历所有节点，所以不需要终止条件
                使用seen记录树枝上使用过的元素，保证不重复使用 """
        path = []
        res = []

        def backtracking(startInd):
            if len(path) >= 2:
                res.append(path[:])     # 因为要遍历所有节点，不需要终止条件
            seen = set()                # 只负责本层递归
            for i in range(startInd, len(nums)):
                if (len(path) and path[-1] > nums[i]) or (nums[i] in seen):
                    continue
                seen.add(nums[i])       # 不需要pop
                path.append(nums[i])
                backtracking(i + 1)
                path.pop()

        backtracking(0)
        return res


if __name__ == '__main__':
    sl = Solution()

    nums = [4,6,7,7]
    print(sl.findSubsequences(nums))
