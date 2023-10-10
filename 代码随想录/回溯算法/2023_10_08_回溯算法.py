from typing import List


class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        """ 77.组合
            中等 """
        # 时间：O(n*2^n)   空间：O(n)
        # path = []
        # res = []
        #
        # def backtracking(n, k, startInd):
        #     if len(path) == k:
        #         res.append(path[:])
        #         return
        #     for i in range(startInd, n + 1):
        #         path.append(i)
        #         backtracking(n, k, i + 1)
        #         path.pop()
        #
        # backtracking(n, k, 1)
        # return res

        # 剪枝
        # 时间：O(n*2^n)   空间：O(n)
        path = []
        res = []
        def backtracking(n, k, startInd):
            if len(path) == k:
                res.append(path[:])
                return
            for i in range(startInd, n - (k - len(path)) + 1 + 1):
                path.append(i)
                backtracking(n, k, i + 1)
                path.pop()

        backtracking(n, k, 1)
        return res

    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        """ 216.组合总和III
            中等 """
        # 简单套用回溯三部曲
        # path = []
        # res = []
        #
        # def backtracking(startInd):
        #     if sum(path) == n and len(path) == k:
        #         res.append(path[:])
        #         return
        #     for i in range(startInd, 9 + 1):
        #         path.append(i)
        #         backtracking(i + 1)
        #         path.pop()
        #
        # backtracking(1)
        # return res

        # 剪枝优化
        # 时间：O(n*2^n)   空间：O(n)
        path = []
        res = []

        def backtracking(startInd):
            if len(path) == k:
                if sum(path) == n:
                    res.append(path[:])
                return
            for i in range(startInd, 9 - (k - len(path)) + 1 + 1):      # 针对k剪枝
                path.append(i)
                if sum(path) > n:                                       # 针对n剪枝
                    path.pop()      # 剪枝之前先做回溯
                    return
                backtracking(i + 1)
                path.pop()

        backtracking(1)
        return res

    def letterCombinations(self, digits: str) -> List[str]:
        """ 17.电话号码的字母组合
            中等
            上面题目是对 '一个集合' 求组合，本题是对 '多个集合' 求组合 """
        path = ""
        res = []
        digit2letter = {
                        '2': 'abc', '3': 'def',
            '4': 'ghi', '5': 'jkl', '6': 'mno',
            '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
        }
        if len(digits) == 0:
            return res

        def backtracking(path, index):
            if index == len(digits):
                res.append(path[:])
                return
            letters = digit2letter[digits[index]]
            for i in range(0, len(letters)):
                path += letters[i]
                backtracking(path, index + 1)         # 上面对一个集合求组合时是 i+1, 本题对多个集合求组合是 index+1
                path = path[:-1][:]

        backtracking(path, 0)
        return res

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        """ 39.组合总和
            中等
            求组合问题：
                1.一个集合求组合，要使用startInd
                2.多个集合求组合，不使用startInd
            可重复取，递归函数的参数依然为i，表示可以重复取当前的数 """
        # path = []
        # res = []
        #
        # def backtracking(startInd):
        #     if sum(path) >= target:
        #         if sum(path) == target:
        #             res.append(path[:])
        #         return
        #     for i in range(startInd, len(candidates)):
        #         path.append(candidates[i])
        #         backtracking(i)
        #         path.pop()
        #
        # backtracking(0)
        # return res

        # 剪枝优化
        # 通过sum(path)剪枝，符合剪枝直接return，因为提前nums.sort()了
        path = []
        res = []

        def backtracking(startInd):
            if sum(path) == target:
                res.append(path[:])
                return
            for i in range(startInd, len(candidates)):
                if sum(path) + candidates[i] > target:
                    return
                path.append(candidates[i])
                backtracking(i)
                path.pop()

        candidates.sort()
        backtracking(0)
        return res

    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        """ 40.组合总和II
            一个集合求组合，递归函数使用startInd
            元素不能重复使用，则递归函数参数为 i+1
        39.组合总和 不包含重复元素，递归参数 i+1 即可
        本题难点：candidates包含重复元素，且不能有重复结果——同层去重，则必须先排序 """
        # 已针对target剪枝，剪枝需放在单层搜索中
        # 时间：O(n*2^n)   空间：O(n)
        path = []
        res = []
        used = [False] * len(candidates)

        def backtracking(startInd, used):
            if sum(path) == target:
                res.append(path[:])
                return
            for i in range(startInd, len(candidates)):
                if i > 0 and candidates[i - 1] == candidates[i] and used[i - 1] == False:
                    continue
                if sum(path) + candidates[i] > target:
                    return
                used[i] = True
                path.append(candidates[i])
                backtracking(i + 1, used)
                path.pop()
                used[i] = False

        candidates.sort()
        backtracking(0, used)
        return res

    def partition(self, s: str) -> List[List[str]]:
        """ 131.分割回文串
            中等
            分割问题 类似 组合问题，解题逻辑有一点点绕
            切割对象是一个，使用startInd保证不重复切 """
        # 时间：O(n*2^n)   空间：O(n^2) 没明白
        path = []
        res = []

        def isPal(s):
            """ 判断是否回文串 """
            i, j = 0, len(s) - 1
            while i <= j:
                if s[i] != s[j]:
                    return False
                i += 1
                j -= 1
            return True

        def backtracking(startInd):
            if startInd == len(s):
                res.append(path[:])
                return
            for i in range(startInd, len(s)):
                if not isPal(s[startInd:i + 1]):
                    continue
                path.append(s[startInd:i + 1])
                backtracking(i + 1)
                path.pop()

        backtracking(0)
        return res

    def restoreIpAddresses(self, s: str) -> List[str]:
        """ 93.复原IP地址
            中等
            分割问题
            分割对象是一个集合，使用startInd防止重复切割 """
        path = []
        res = []

        def backtracking(startInd):
            if startInd == len(s) and len(path) == 4:
                res.append(".".join(path))
                return
            for i in range(startInd, len(s)):
                if (len(s[startInd:i + 1]) > 1 and s[startInd] == '0') or len(path) >= 4:
                    continue
                if 0 <= int(s[startInd:i + 1]) <= 255:
                    path.append(s[startInd:i + 1])
                    backtracking(i + 1)
                    path.pop()

        backtracking(0)
        return res

    def subsets(self, nums: List[int]) -> List[List[int]]:
        """ 78.子集
            中等
            上面 组合、切割 问题，是收取树的叶子节点
            本题 子集 问题，是收取所有节点，res收集结果的时候不需要任何条件 """
        # 时间：O(n*2^n) 每个节点都有取/不取2种状态 构建每种状态需要O(n)   空间：O(n)递归深度
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
            同层剪枝 """
        # 时间：O(n*2^n)   空间：O(n)
        path = []
        res = []
        used = [False] * len(nums)

        def backtracking(startInd):
            res.append(path[:])
            for i in range(startInd, len(nums)):
                if i > 0 and nums[i - 1] == nums[i] and used[i - 1] == False:
                    continue
                used[i] = True
                path.append(nums[i])
                backtracking(i + 1)
                path.pop()
                used[i] = False

        nums.sort()
        backtracking(0)
        return res

    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        """ 491.递增子序列
            中等
            类似子集问题，但不是收集所有节点，而是收集len(path)>1的节点 """
        # 时间：O(n*2^n)   空间：O(n)
        path = []
        res = []

        def backtracking(startInd):
            if len(path) >= 2:
                res.append(path[:])
            seen = set()
            for i in range(startInd, len(nums)):
                if nums[i] in seen or (len(path) and path[-1] > nums[i]):
                    continue
                seen.add(nums[i])
                path.append(nums[i])
                backtracking(i + 1)
                path.pop()

        backtracking(0)
        return res


if __name__ == "__main__":
    sl = Solution()

    nums = [4,4,3,2,1]
    print(sl.findSubsequences(nums))

    """
    回溯三部曲：
        1.函数参数
        2.终止条件
        3.单层搜索
    """