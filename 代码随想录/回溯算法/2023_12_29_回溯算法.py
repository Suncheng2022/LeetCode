from typing import List


class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        """ 77.组合 """
        # path = []
        # res = []
        #
        # def backtracking(startInd):
        #     # 终止条件
        #     if len(path) == k:
        #         res.append(path[:])
        #         return
        #     # 单层递归
        #     for i in range(startInd, n + 1):
        #         path.append(i)
        #         backtracking(i + 1)
        #         path.pop()
        #
        # backtracking(1)
        # return res

        # 回溯是暴力搜索，至多也就剪枝优化一下啦
        # 不认真，调错耗时
        path = []
        res = []

        def backtracking(startInd):
            # 终止条件
            if len(path) == k:
                res.append(path[:])
                return
            # 单层递归
            for i in range(startInd, n - (k - len(path)) + 1 + 1):      # for循环的起点的范围
                path.append(i)
                backtracking(i + 1)
                path.pop()

        backtracking(1)
        return res

    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        """ 216.组合总和III """
        # path = []
        # res = []
        #
        # def backtracking(startInd):
        #     # 终止条件
        #     if sum(path) == n and len(path) == k:
        #         res.append(path[:])
        #         return
        #     # 单层递归
        #     for i in range(startInd, 9 + 1):       # 数的范围，1~9
        #         path.append(i)
        #         backtracking(i + 1)
        #         path.pop()
        #
        # backtracking(1)
        # return res

        # 当然可以剪枝
        # path = []
        # res = []
        #
        # def backtracking(startInd):
        #     # 终止条件
        #     if len(path) == k:
        #         if sum(path) == n:
        #             res.append(path[:])
        #         return
        #     # 单层递归
        #     for i in range(startInd, 9 + 1):        # 数的范围 1~9
        #         if sum(path) + i > n:               # 剪枝_法一
        #             return
        #         path.append(i)
        #         backtracking(i + 1)
        #         path.pop()
        #
        # backtracking(1)
        # return res

        # 再剪枝
        path = []
        res = []

        def backtracking(startInd):
            # 终止条件
            if len(path) == k:
                if sum(path) == n:
                    res.append(path[:])
                return
            # 单层递归
            for i in range(startInd, 9 - (k - len(path)) + 1 + 1):      # 剪枝_法二 这两次加一的原因你可知道——1次是起始位置，1次是为取到
                path.append(i)
                backtracking(i + 1)
                path.pop()

        backtracking(1)
        return res

    def letterCombinations(self, digits: str) -> List[str]:
        """ 17.电话号码的字母组合
            上面题目是一个集合求组合，本题是多个集合求组合 """
        path = []
        res = []
        digit2letter = {2: 'abc', 3: 'def',
                        4: 'ghi', 5: 'jkl', 6: 'mno',
                        7: 'pqrs', 8: 'tuv', 9: 'wxyz'}

        def backtracking(index):            # 此index指示按键对应的字母集，不同于上面题目的startInd
            # 终止条件
            if index == len(digits):        # 处理完了所有digits；index指示正在处理的第index个按键字母集
                res.append("".join(path[:]))
                return
            # 单层递归
            letters = digit2letter[int(digits[index])]
            for c in letters:
                path.append(c)
                backtracking(index + 1)
                path.pop()

        if len(digits) == 0:
            return []
        backtracking(0)
        return res

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        """ 39.组合总和
            组合问题：
                1.一个集合求组合，使用startIndex
                2.多个集合求组合，不使用startIndex
                3.元素可重复使用，递归函数参数startIndex不加1 """
        # path = []
        # res = []
        #
        # def backtracking(startInd):
        #     # 终止条件
        #     if sum(path) > target:
        #         return
        #     elif sum(path) == target:
        #         res.append(path[:])
        #         return
        #     # 单层递归
        #     for i in range(startInd, len(candidates)):
        #         path.append(candidates[i])
        #         backtracking(i)                  # 参数startInd不加1，意味着元素可重复选取
        #         path.pop()
        #
        # backtracking(0)
        # return res

        # 剪枝：如果已知和>target了，则本层循环后面都不考虑了(意味着数组有序，对不)
        path = []
        res = []

        def backtracking(startIndex):               # 求组合：一个集合求组合使用startIndex，多个集合求组合不使用startIndex
            # 终止条件 剪枝，不用判断>target了
            if sum(path) == target:
                res.append(path[:])
                return
            # 单层递归
            for i in range(startIndex, len(candidates)):
                if sum(path) + candidates[i] > target:
                    return                          # 剪枝，超过则终止
                path.append(candidates[i])
                backtracking(i)                     # 这里不止一次犯过错
                path.pop()

        candidates.sort()
        backtracking(0)
        return res

    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        """ 40.组合总和II
                39.组合总和 无重复元素，且可重复使用，不能包含重复的结果
                40.组合总和II 有重复元素，只能使用一次，且不能包含重复的结果-->得“去重”了
            本题并不简单，因为有重复元素，结果还要求不能有重复组合——得‘去重’了 """
        path = []
        res = []
        used = [False] * len(candidates)

        def backtracking(startIndex):
            # 终止条件
            if sum(path) == target:
                res.append(path[:])
                return
            # 单层递归
            for i in range(startIndex, len(candidates)):
                if sum(path) + candidates[i] > target:      # 必须剪枝，否则超时
                    return
                if i > 0 and candidates[i] == candidates[i - 1] and used[i - 1] == False:
                    continue                                # 为什么是continue而不是return，因为仅仅是去重，使用过的元素不再重复使用了，并不是要结束本层for循环
                path.append(candidates[i])
                used[i] = True
                backtracking(i + 1)                         # 元素只能使用一次
                path.pop()
                used[i] = False

        candidates.sort()
        backtracking(0)
        return res

    def partition(self, s: str) -> List[List[str]]:
        """ 131.分割回文串
            1.切割问题 类似 组合问题
            2.startIndex就是切割线 """
        path = []
        res = []

        def backtracking(startIndex):
            # 终止条件
            if startIndex == len(s):
                res.append(path[:])
                return
            # 单层递归
            for i in range(startIndex, len(s)):
                if isPalind(s[startIndex:i + 1]):
                    path.append(s[startIndex:i + 1][:])
                    backtracking(i + 1)
                    path.pop()

        def isPalind(s):
            if len(s) <= 1:
                return True
            l, r = 0, len(s) - 1
            while l <= r:
                if s[l] != s[r]:
                    return False
                l += 1
                r -= 1
            return True

        backtracking(0)
        return res

    def restoreIpAddresses(self, s: str) -> List[str]:
        """ 93.复原IP地址
            切割问题 类似 组合问题 """
        path = []
        res = []

        def backtracking(startIndex):               # startIndex就是切割点
            # 终止条件
            if startIndex == len(s) and len(path) == 4:
                res.append(".".join(path[:]))
                return
            # 单层递归
            for i in range(startIndex, len(s)):
                tmp = s[startIndex:i + 1]
                if (len(tmp) > 1 and tmp[0] == '0') or int(tmp) < 0 or int(tmp) > 255:
                    continue
                path.append(tmp)
                backtracking(i + 1)
                path.pop()

        backtracking(0)
        return res

    def subsets(self, nums: List[int]) -> List[List[int]]:
        """ 78.子集
            子集问题 不同于 组合问题、切割问题
            1.组合问题、切割问题 收集所有叶子节点
            2.子集问题 收集所有节点 """
        path = []
        res = []

        def backtracking(startIndex):
            # 终止条件
            res.append(path[:])
            # 单层递归
            for i in range(startIndex, len(nums)):
                path.append(nums[i])
                backtracking(i + 1)
                path.pop()

        backtracking(0)
        return res

    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        """ 90.子集II
            相比 78.子集，本题有重复元素，但结果要求无重复，所以得‘去重’ """
        path = []
        res = []
        used = [False] * len(nums)

        def backtracking(startIndex):
            # 终止条件
            res.append(path[:])
            # 单层递归
            for i in range(startIndex, len(nums)):
                if i > 0 and nums[i] == nums[i - 1] and used[i - 1] == False:
                    continue                                                    # continue而不是return，本层后面的还是要遍历的
                path.append(nums[i])
                used[i] = True
                backtracking(i + 1)
                used[i] = False
                path.pop()

        nums.sort()
        backtracking(0)
        return res

    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        """ 491.非递减子序列 递增子序列
            求 非递减子序列 类似 求子集，遍历所有节点
            求 非递减子序列，去重不能对数组排序，每层for都记录下seen，用过的元素不重复用 """
        path = []
        res = []

        def backtracking(startIndex):
            # 终止条件
            if len(path) >= 2:
                res.append(path[:])             # 相当于不加终止条件，类似 子集 那样遍历所有节点
            # 单层递归
            seen = set()                        # 求 递增子序列，去重不能对数组排序，那就每层for循环时记录一下访问的节点，用过的别重复用
            for i in range(startIndex, len(nums)):
                if nums[i] not in seen and (not len(path) or path[-1] <= nums[i]):
                    seen.add(nums[i])           # 【不参加回溯】
                    path.append(nums[i])
                    backtracking(i + 1)
                    path.pop()

        backtracking(0)
        return res

    def permute(self, nums: List[int]) -> List[List[int]]:
        """ 46.全排列
            不含重复元素
            排列问题：
                1.与 组合、子集 的不同，就是不使用startIndex了，因为——如1在[1,2]中使用后在[2,1]中还要再使用
                2.使用used数组标记path中使用了哪些元素 """
        path = []
        res = []
        used = [False] * len(nums)

        def backtracking():
            # 终止条件
            if len(path) == len(nums):
                res.append(path[:])
                return
            # 单层递归
            for i in range(len(nums)):
                if not used[i]:
                    path.append(nums[i])
                    used[i] = True
                    backtracking()
                    used[i] = False
                    path.pop()

        backtracking()
        return res

    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        """ 47.全排列II
            有重复元素
            排列问题：
                不使用startIndex，但要使用used记录path中使用了哪些元素-->本题元素有重复，需要从 层 角度去重，因此同时使用used完成这2个功能。"""
        path = []
        res = []
        used = [False] * len(nums)

        def backtracking():
            # 终止条件
            if len(path) == len(nums):
                res.append(path[:])
                return
            # 单层递归
            for i in range(len(nums)):
                if used[i] or (i > 0 and nums[i - 1] == nums[i] and used[i - 1] == False):
                    continue
                path.append(nums[i])
                used[i] = True
                backtracking()
                used[i] = False
                path.pop()

        nums.sort()
        backtracking()
        return res