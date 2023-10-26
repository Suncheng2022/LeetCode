from typing import List


class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        """ 216.组合总和III
            回溯三部曲：
                1.递归函数参数和返回值
                2.终止条件
                3.单层搜索 """
        path = []
        res = []

        def backtracking(startInd):
            if len(path) == k:
                if sum(path) == n:
                    res.append(path[:])
                return
            for i in range(startInd, 9 + 1):    # 从 1~9 搜索
                if sum(path) > n:               # 剪枝
                    return
                path.append(i)
                backtracking(i + 1)
                path.pop()

        backtracking(1)
        return res

    def letterCombinations(self, digits: str) -> List[str]:
        """ 17.电话号码的字母组合
            上面的题目是求 一个集合 的组合，本题是求 多个集合 的组合，因此不使用startInd -> 而是使用index记录正在处理的数字即可 """
        digit2letter = {'2': 'abc', '3': 'def',
                        '4': 'ghi', '5': 'jkl', '6': 'mno',
                        '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}
        path = []
        res = []

        def backtracking(index):
            """ 求多个集合的组合，使用index而不是startInd """
            if index == len(digits):
                res.append("".join(path))
                return
            letters = digit2letter[digits[index]]
            for c in letters:
                path.append(c)
                backtracking(index + 1)
                path.pop()
        if len(digits) == 0:
            return res
        backtracking(0)
        return res

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        """ 39.组合总和
            组合问题：1.对 一个集合 求组合，使用startInd
                    2.对 多个集合 求组合，不使用startInd
            元素可重复选取，递归函数的参数为 i 而不是 i+1 """
        path = []
        res = []

        def backtracking(startInd):     # 本题是 一个集合 求组合，所以使用startInd
            # 终止条件
            if sum(path) == target:
                res.append(path[:])
                return
            # 单层搜索
            for i in range(startInd, len(candidates)):
                if sum(path) > target:  # 其实也是终止条件
                    return
                path.append(candidates[i])
                backtracking(i)         # 元素可重复使用，递归函数的参数为 i，而不是 i + 1
                path.pop()

        backtracking(0)
        return res

    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        """ 40.组合总和II
            组合问题：
                1.对 一个集合 求组合，使用startInd
                2.对 多个集合 求组合，不使用startInd
            元素不能重复使用，递归函数参数为 i+1
            去重：本题有重复元素，还需要结果去重，使用used对同层使用过的元素去重 """
        used = [False] * len(candidates)
        path = []
        res = []

        def backtracking(startInd):
            # 终止条件
            if sum(path) == target:
                res.append(path[:])
                return
            # 单层搜索
            for i in range(startInd, len(candidates)):
                if sum(path) + candidates[i] > target:      # 剪枝，就是后面都不要了 即return；不剪枝会超时
                    return
                if i > 0 and candidates[i - 1] == candidates[i] and used[i - 1] == False:
                    continue
                path.append(candidates[i])
                used[i] = True
                backtracking(i + 1)
                path.pop()
                used[i] = False

        candidates.sort()
        backtracking(0)
        return res

    def partition(self, s: str) -> List[List[str]]:
        """ 131.分割回文串
            分割问题 == 组合问题
                        1.一个集合 求组合，使用startInd
                        2.多个集合 求组合，不使用startInd"""
        def is_pal(s):
            l, r = 0, len(s) - 1
            while l <= r and s[l] == s[r]:
                l += 1
                r -= 1
            return False if l <= r else True

        res = []
        path = []

        def backtracking(startInd):
            # 终止条件
            if startInd == len(s):
                res.append(path[:])
                return
            # 单层搜索
            for i in range(startInd, len(s)):
                if is_pal(s[startInd:i + 1]):
                    path.append(s[startInd:i + 1])
                else:
                    continue
                backtracking(i + 1)
                path.pop()

        backtracking(0)
        return res

    def restoreIpAddresses(self, s: str) -> List[str]:
        """ 93.复原IP地址
            切割问题 == 组合问题：
                1.一个集合 使用startInd
                2.多个集合 不使用startInd
            终止条件相比切割稍变化了一点"""
        path = []
        res = []

        def backtracking(startInd):
            # 终止条件(不再是分割点到了字符串最后，而是分割为4段就终止)
            if len(path) == 4:
                res.append(".".join(path))
                return
            # 单层搜索
            for i in range(startInd, len(s)):
                if (len(s[startInd:i + 1]) > 1 and s[startInd] == '0') or (not 0 <= int(s[startInd:i + 1]) <= 255):
                    continue
                path.append(s[startInd:i + 1])
                backtracking(i + 1)
                path.pop()

        backtracking(0)
        return res

    def subsets(self, nums: List[int]) -> List[List[int]]:
        """ 78.子集 无重复元素
            组合问题、分割问题——收集叶节点，子集问题——收集所有节点；子集 == 组合
            组合问题：
                1.对 一个集合 求组合，使用startInd
                2.对 多个集合 求组合，不使用startInd
                ->3.子集无序，所以不重复取，使用startInd
            子集问题 vs 组合问题：
                组合 收集 叶节点
                子集 收集 所有节点
            子集问题，不需要剪枝，就是要遍历所有节点 """
        path = []
        res = []

        def backtracking(startInd):
            # 终止条件
            res.append(path[:])
            # 单层搜索
            for i in range(startInd, len(nums)):
                path.append(nums[i])
                backtracking(i + 1)
                path.pop()

        backtracking(0)
        return res

    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        """ 90.子集II 有重复元素
            子集问题 == 组合问题：
                1.对 一个集合 求组合，使用startInd
                2.对 多个集合 求组合，不使用startInd
                -> 3.子集结果无序，因此不重复使用，使用startInd
            子集问题 vs 组合问题：
                组合问题 收集 叶节点
                子集问题 收集 所有节点
            子集问题 不需要剪枝 就是要遍历所有节点
            去重，排序 + used 同层去重 """
        path = []
        res = []
        used = [False] * len(nums)

        def backtracking(startInd, used):
            # 终止条件
            res.append(path[:])
            # 单层搜索
            for i in range(startInd, len(nums)):
                if i > 0 and nums[i - 1] == nums[i] and used[i - 1] == False:       # 同层去重
                    continue
                path.append(nums[i])
                used[i] = True
                backtracking(i + 1, used)
                path.pop()
                used[i] = False

        nums.sort()
        backtracking(0, used)
        return res