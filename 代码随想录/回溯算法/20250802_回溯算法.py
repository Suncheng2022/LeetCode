"""
08.02   将近一月, 信心浮现
"""
from typing import List

class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        """ 77.组合 """
        # 时间:O(n^2) 空间:O(logn)
        # path = []
        # res = []

        # def backtrack(n, k, startInd):
        #     nonlocal path, res
        #     # 终止条件
        #     if len(path) == k:
        #         res.append(path[:])
        #         return
        #     # 回溯
        #     for i in range(startInd, n + 1):
        #         path.append(i)
        #         backtrack(n, k, i + 1)
        #         path.pop()
        
        # backtrack(n, k, 1)
        # return res

        ## 剪枝优化
        res = []
        path = []
        def backtrack(n, k, startInd):
            nonlocal res, path
            if len(path) == k:
                res.append(path[:])
                return
            for i in range(startInd, n - (k - len(path)) + 1 + 1):
                path.append(i)
                backtrack(n, k, i + 1)
                path.pop()
        backtrack(n, k, 1)
        return res
    
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        """ 216.组合总和III """
        ## 完美版
        path = []
        res = []
        def backtrack(k, n, startInd):
            if sum(path) > n:       # 剪枝
                return
            if len(path) == k:
                if sum(path) == n:
                    res.append(path[:])
                return
            for i in range(startInd, 9 - (k - len(path)) + 1 + 1):  # 剪枝
                path.append(i)
                backtrack(k, n, i + 1)
                path.pop()

        backtrack(k, n, 1)
        return res
    
    def letterCombinations(self, digits: str) -> List[str]:
        """ 17.电话号码的字母组合 \n
            求多个集合的组合 """
        d2l = {k:v for k, v in zip([str(x) for x in range(2, 10)], ['abc', 'def', 'ghi', 'jkl', 'mno', 'pqrs', 'tuv', 'wxyz'])}
        path = []
        res = []
        def backtrack(digits, index):
            # 终止条件
            if index == len(digits):
                res.append(''.join(path))
                return
            # 单层搜索逻辑
            letters = d2l[digits[index]]
            for lt in letters:
                path.append(lt)
                backtrack(digits, index + 1)
                path.pop()
        
        if len(digits) == 0:
            return res
        backtrack(digits, 0)
        return res
    
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        """ 39.组合总和 \n
            本题 元素可重复选取, 则startInd不用+1 \n
            在 一个集合 中求组合, 就需要用startInd. 在 多个集合 中求组合, 不需要用startInd. """
        # res = []
        # path = []
        # def backtrack(candidates, startInd):
        #     if sum(path) > target:
        #         return
        #     elif sum(path) == target:
        #         res.append(path[:])
        #         return
        #     for i in range(startInd, len(candidates)):
        #         path.append(candidates[i])
        #         backtrack(candidates, i)
        #         path.pop()
        
        # backtrack(candidates, 0)
        # return res

        ## 剪枝优化
        path = []
        res = []
        def backtrack(candidates, startInd):
            if sum(path) > target:
                return
            elif sum(path) == target:
                res.append(path[:])
                return
            for i in range(startInd, len(candidates)):
                if sum(path) + candidates[i] > target:      # candidates已排序, 这一层for后面不用试了
                    break                                   # break/return 在此处表达的意思是一样的
                path.append(candidates[i])
                backtrack(candidates, i)
                path.pop()
        
        candidates.sort()
        backtrack(candidates, 0)
        return res
    
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        """ 40.组合总和II \n
            与 39.组合总和 差别很大! """
        # res = []
        # path = []
        # used = [False] * len(candidates)
        # def backtrack(used, startInd):
        #     if sum(path) == target:
        #         res.append(path[:])
        #         return
        #     for i in range(startInd, len(candidates)):
        #         if i and candidates[i] == candidates[i - 1] and used[i - 1] == False:   # 同层使用过, 用于同层去重
        #             continue
        #         if sum(path) + candidates[i] > target:
        #             break
        #         path.append(candidates[i])
        #         used[i] = True
        #         backtrack(used, i + 1)
        #         used[i] = False
        #         path.pop()
        
        # candidates.sort()
        # backtrack(used, 0)
        # return res

        ## 不使用used去重
        res = []
        path = []
        def backtrack(startInd):
            if sum(path) == target:
                res.append(path[:])
                return
            for i in range(startInd, len(candidates)):
                if i > startInd and candidates[i] == candidates[i - 1]:     # 同层不使用一样的, 用于同层去重
                    continue
                if sum(path) + candidates[i] > target:
                    break
                path.append(candidates[i])
                backtrack(i + 1)
                path.pop()
        
        candidates.sort()
        backtrack(0)
        return res
    
    def partition(self, s: str) -> List[List[str]]:
        """ 131.分割回文串 \n
            类似 组合问题"""
        def isValid(s: str) -> bool:
            i, j = 0, len(s) - 1
            while i <= j:
                if s[i] != s[j]:
                    return False
                i += 1
                j -= 1
            return True
        
        res = []
        path = []
        def backtrack(startInd):
            # 终止条件
            if startInd >= len(s):
                res.append(path[:])
                return
            for i in range(startInd, len(s)):
                if isValid(s[startInd:i + 1]):
                    path.append(s[startInd:i + 1])
                    backtrack(i + 1)
                    path.pop()
        
        backtrack(0)
        return res

    def restoreIpAddresses(self, s: str) -> List[str]:
        """ 93.复原IP地址 """
        ## 自己写出来了, 给自己一个nb
        def isVal(s):
            if len(s) > 1 and s[0] == '0':
                return False
            elif not 0 <= int(s) <= 255:
                return False
            return True

        res = []
        path = []
        def backtrack(startInd):
            if startInd >= len(s):
                if len(path) == 4:
                    res.append(".".join(path[:]))
                return
            for i in range(startInd, len(s)):
                if not isVal(s[startInd:i + 1]):
                    continue
                path.append(s[startInd:i + 1])
                backtrack(i + 1)
                path.pop()
        backtrack(0)
        return res
    
    def subsets(self, nums: List[int]) -> List[List[int]]:
        """ 78.子集 \n
            不同于 组合问题/分割问题收集叶节点, 子集问题是收集所有节点 """
        res = []
        path = []
        def backtrack(startInd):
            res.append(path[:])
            if startInd >= len(nums):
                return
            for i in range(startInd, len(nums)):
                path.append(nums[i])
                backtrack(i + 1)
                path.pop()
        backtrack(0)
        return res
    
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        """ 90.子集II \n
            有重复元素, 结果不能有重复, 那么就是同层去重 """
        res = []
        path = []
        used = [False] * (len(nums))
        
        def backtrack(startInd):
            res.append(path[:])
            for i in range(startInd, len(nums)):
                if i and nums[i] == nums[i - 1] and used[i - 1] == False:
                    continue
                path.append(nums[i])
                used[i] = True
                backtrack(i + 1)
                used[i] = False
                path.pop()
        
        nums.sort()
        backtrack(0)
        return res

    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        """ 491.非递减子序列 """
        res = []
        path = []

        def backtrack(startInd):
            if len(path) >= 2:
                res.append(path[:])
            used = set()
            for i in range(startInd, len(nums)):
                if (path and path[-1] > nums[i]) or nums[i] in used:
                    continue
                path.append(nums[i])
                used.add(nums[i])
                backtrack(i + 1)
                path.pop()
        
        backtrack(0)
        return res
    
    def permute(self, nums: List[int]) -> List[List[int]]:
        """ 46.全排列 """
        res = []
        path = []
        used = [False] * len(nums)
        
        def backtrack():
            if len(path) == len(nums):
                res.append(path[:])
            for i in range(len(nums)):
                if used[i]:
                    continue
                path.append(nums[i])
                used[i] = True
                backtrack()
                used[i] = False
                path.pop()
        
        backtrack()
        return res
    
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        """ 47.全排列II \n
            有重复元素, 不能有重复结果 同层去重 """
        res = []
        path = []
        used = [False] * len(nums)

        def backtrack():
            if len(path) == len(nums):
                res.append(path[:])
            for i in range(len(nums)):
                if i and nums[i] == nums[i - 1] and used[i - 1] == False:
                    continue
                if used[i]:
                    continue
                path.append(nums[i])
                used[i] = True
                backtrack()
                used[i] = False
                path.pop()
        
        nums.sort()
        backtrack()
        return res

if __name__ == '__main__':
    """
    回溯三部曲:
        1.确定递归函数参数和返回值
        2.回溯函数终止条件
        3.单层搜索逻辑
    """
    sl = Solution()
    nums = [1,1,2]
    print(sl.permuteUnique(nums))