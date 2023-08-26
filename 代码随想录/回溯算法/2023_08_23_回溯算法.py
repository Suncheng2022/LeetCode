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
        def isValid(s):
            if len(s) == 0 or (len(s) > 1 and s[0] == '0') or not 0 <= int(s) <= 255:
                return False
            return True
        def backtracking(startInd):
            if startInd == len(s) and len(path) == 4:
                res.append(".".join(path))
                return
            for i in range(startInd, len(s)):
                if not isValid(s[startInd:i + 1]):
                    continue
                path.append(s[startInd:i + 1][:])
                backtracking(i + 1)
                path.pop()

        res = []
        path = []
        backtracking(0)
        return res

    def partition(self, s: str) -> List[List[str]]:
        """ 131.分割回文串 """
        def isValid(s):
            l, r = 0, len(s) - 1
            while l <= r:
                if s[l] != s[r]:
                    return False
                l += 1
                r -= 1
            return True

        def backtracking(startInd):
            if startInd >= len(s):
                res.append(path[:])
                return
            for i in range(startInd, len(s)):
                if not isValid(s[startInd:i + 1]):
                    continue
                path.append(s[startInd:i + 1][:])
                backtracking(i + 1)
                path.pop()

        res = []
        path = []
        backtracking(0)
        return res

    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        """ 40.数组总和II """
        def backtracking(startInd):
            if sum(path) == target:
                res.append(path[:])
                return
            for i in range(startInd, len(candidates)):
                if i > 0 and candidates[i - 1] == candidates[i] and not used[i - 1]:
                    continue
                if sum(path) > target:
                    break
                path.append(candidates[i])
                used[i] = True
                backtracking(i + 1)
                path.pop()
                used[i] = False

        res = []
        path = []
        used = [False] * len(candidates)
        candidates.sort()
        backtracking(0)
        return res

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        """ 39.数组总和 """
        def backtracking(startInd):
            if sum(path) == target:
                res.append(path[:])
                return
            for i in range(startInd, len(candidates)):
                if sum(path) + candidates[i] > target:      # 剪枝，而不应该只写 sum(path) > target.
                    break
                path.append(candidates[i])
                backtracking(i)
                path.pop()

        res = []
        path = []
        backtracking(0)
        return res

    def letterCombinations(self, digits: str) -> List[str]:
        """ 17.电话号码的字母组合
            回溯 模拟多层for循环
            多个集合求组合，不使用startIndex
            backtracking(index)的参数用来指示当前处理digits的第几个字符 """
        if not len(digits):
            return []
        digit2letter = dict(zip(range(2, 10), ['abc', 'def',
                                               'ghi', 'jkl', 'mno',
                                               'pqrs', 'tuv', 'wxyz']))

        def backtracking(index, path):
            if index == len(digits):
                res.append(path[:])
                return
            letters = digit2letter[int(digits[index])]
            for i in range(0, len(letters)):
                path += letters[i]
                backtracking(index + 1, path)
                path = path[:-1]

        res = []
        path = ""
        backtracking(0, path)
        return res

    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        """ 216.数组总和III
            递归参数：想一下回溯的树结构，同一层 不使用重复元素，否则产生相同结果
            终止条件：k个数
            遍历：就是for i in range(startInd, 10)  范围固定1~9
            """
        def backtracking(startInd):
            if len(path) == k:
                if sum(path) == n:
                    res.append(path[:])
                return
            for i in range(startInd, 10):
                if sum(path) + i > n:
                    break
                path.append(i)
                backtracking(i + 1)
                path.pop()

        res = []
        path = []
        backtracking(1)
        return res

    def combine(self, n: int, k: int) -> List[List[int]]:
        """ 77.组合
            一个集合求组合，使用startIndex """
        def backtracking(startInd):
            if len(path) == k:
                res.append(path[:])
                return
            for i in range(startInd, n + 1):
                # if len(path) + 1 > k:   # 这个剪枝效率太低
                #     break
                # 剪枝，效率更高
                if i > n - (k - len(path)) + 1:     # 边界条件需要细想一下
                    break
                path.append(i)
                backtracking(i + 1)
                path.pop()

        res = []
        path = []
        backtracking(1)
        return res


if __name__ == '__main__':
    sl = Solution()

    k = 2
    n = 4
    print(sl.combine(n, k))