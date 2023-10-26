from typing import List


class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        """ 216.数组总和III
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