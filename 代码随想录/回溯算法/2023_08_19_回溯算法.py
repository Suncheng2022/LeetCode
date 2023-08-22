""" 复习 回溯算法 《代码随想录》 """
from typing import List


class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        """ 77.组合
            本题求组合——组合没有顺序，排列有顺序。"""
        def backtracking(startInd):
            # 回溯递归的终止条件
            if len(path) == k:
                res.append(path[:])
                return
            # 遍历 属性结构的 “层”
            for i in range(startInd, n + 1):    # 因为组合没有顺序，所以同一“层”不能选取相同的元素; 回溯树形结构的每“层”从哪个元素开始遍历，靠的就是startInd
                # 剪枝操作，如果某“层”遍历起始元素的索引已不足构成k个数，就剪掉
                if i <= n - (k - len(path)) + 1:
                    path.append(i)
                    backtracking(i + 1)
                    path.pop()
                else:
                    break

        res = []
        path = []
        backtracking(1)
        return res

    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        """ 216.组合总和III """
        def backtracking(startInd):
            if sum(path) > n:       # 剪枝
                return
            if len(path) == k:
                if sum(path) == n:
                    res.append(path[:])
                return
            for i in range(startInd, 10):
                path.append(i)
                backtracking(i + 1)
                path.pop()

        res = []
        path = []
        backtracking(1)
        return res

    def letterCombinations(self, digits: str) -> List[str]:
        """ 17.电话号码的字母组合 """
        if len(digits) == 0:
            return []
        digits2letters = dict(zip(range(2, 10), ["abc", "def",
                                                 "ghi", 'jkl', 'mno',
                                                 'pqrs', 'tuv', 'wxyz']))

        def backtracking(index, path):    # 不是之前的startIndex，index指示遍历到digits的第几个字符
            if len(path) == len(digits):
                res.append(path[:])
                return
            d = int(digits[index])
            for c in digits2letters[d]:
                path += c
                backtracking(index + 1, path)
                path = path[:-1]

        res = []
        path = ""
        backtracking(0, path)
        return res

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        """ 39.数组总和
            可重复选取
            《代码随想录》：一个集合求组合，就需要startInd；多个集合求组合不需要startInd """
        # def backtracking(startInd):
        #     # 回溯递归终止条件，随递归不断深入，path不断加入元素，sum(path)会慢慢变大，所以超过target的就不要了[若不停止，会不断再深入递归，可重复选取呀]
        #     if sum(path) == target:
        #         res.append(path[:])
        #         return
        #     elif sum(path) > target:        # 这好像不属于剪枝，因为若不及时丢弃sum(path)>target的结果会导致继续往深层递归，报错
        #         return
        #     # 这里需要细品一下，为什么递归参数是 i ——可以思考一下回溯树形图
        #     for i in range(startInd, len(candidates)):      # 一个集合求组合就需要startInd
        #         path.append(candidates[i])
        #         backtracking(i)         # 【注意】允许重复选取，递归参数不是 i+1 而是 i
        #         path.pop()
        #
        # res = []
        # path = []
        # backtracking(0)
        # return res

        # 基于上面的代码 剪枝
        # 剪枝通常先排序？
        def backtracking(startInd):
            # 回溯递归终止条件，随递归不断深入，path不断加入元素，sum(path)会慢慢变大，所以超过target的就不要了[若不停止，会不断再深入递归，可重复选取呀]
            if sum(path) == target:
                res.append(path[:])
                return
            # elif sum(path) > target:        # 这好像不属于剪枝，因为若不及时丢弃sum(path)>target的结果会导致继续往深层递归，报错
            #     return
            # 这里需要细品一下，为什么递归参数是 i ——可以思考一下回溯树形图
            for i in range(startInd, len(candidates)):      # 一个集合求组合就需要startInd
                if sum(path) + candidates[i] > target:      # 剪枝，已排序，只要超过target，就不必继续遍历了
                    break
                path.append(candidates[i])
                backtracking(i)         # 【注意】允许重复选取，递归参数不是 i+1 而是 i
                path.pop()

        res = []
        path = []
        candidates.sort()       # 剪枝 先排序
        backtracking(0)
        return res

    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        """ 40.数组总和II
            本题需要 去重，使用used辅助 同一层遍历是否选取过相同元素，若选取过则continue，而不必把整个“层”break。前面题目剪枝用到sum(path)的时候break了，因为超过了target则后面遍历必然也超过，相当于提前结束无意义遍历
            目前为止：剪枝、去重，都有可能要求排序 """
        def backtracking(startInd):     # 一个集合找组合，要startInd
            if sum(path) == target:
                res.append(path[:])
                return
            for i in range(startInd, len(candidates)):
                # 剪枝，否则会超时——因为去重已经排序了(当然这样剪枝也需要排序)
                if sum(path) + candidates[i] > target:      # 则当前“层”都可以结束了 因为已经排序了
                    break
                # 去重, 当前遍历元素与前一个元素相同(前提是排序了)，且used[i-1]=False即当前元素是通过上一个元素回溯之后才访问，这说明是同层相同元素。
                # 若本条件改为used[i-1]=True，表示一个树枝内刚访问过该元素(candidates中两个相同元素)，并非重复选取相同元素
                if i > 0 and candidates[i - 1] == candidates[i] and not used[i - 1]:
                    continue
                path.append(candidates[i])
                used[i] = True
                backtracking(i + 1)     # 元素只能使用一次；i+1则不剪枝也能得出正确结果
                path.pop()
                used[i] = False

        res = []
        path = []
        used = [False] * len(candidates)
        candidates.sort()       # 本题去重要求这么做
        backtracking(0)
        return res

    def partition(self, s: str) -> List[List[str]]:
        """ 131.分割回文串
            《代码随想录》：分割类似组合问题，所以使用startInd """
        def isPal(s):
            """ 判断s是否为回文串 """
            l, r = 0, len(s) - 1
            while l <= r:
                if s[l] != s[r]:
                    return False
                l += 1
                r -= 1
            return True

        def backtracking(startInd):
            """ startInd:切割起始位置索引(从某字符后开始切割、切割首元素之前，能懂不) """
            if startInd >= len(s):      # 切割起始位置到了字符串最后，表示完成了一种切割方式
                res.append(path[:])
                return
            # 遍历，计算每种切割方式
            for i in range(startInd, len(s)):
                if isPal(s[startInd:i + 1]):        # s[startInd:i + 1]若是回文串
                    path.append(s[startInd:i + 1][:])       # 注意这里是添加到path中；path表示一种切割方式
                else:       # 若某种切割方式，其中切下来的段s[startInd:i + 1]不是回文，说明此种切割方式不符合题目要求，不往后切了
                    continue
                backtracking(i + 1)     # 因为切割不能切割重复元素
                path.pop()

        res = []
        path = []
        backtracking(0)
        return res

    def restoreIpAddresses(self, s: str) -> List[str]:
        """ 93.复原IP地址
            注意 插入 和 删除 '.'的索引
                收集结果时要同时判断最后一段是否合法
            本题也可使用path来收集某种切割的结果，就不用插入和删除'.'了，实现更简洁 """
        # def isVal(s):
        #     """ s是否合法 """
        #     if len(s) == 0 or (len(s) > 1 and s[0] == '0') or not 0 <= int(s) <= 255:
        #         return False
        #     return True
        #
        # def insert(s, ind):
        #     """ 传入字符串s，在索引ind之前插入'.'
        #         返回插入'.'后的字符串s """
        #     s = list(s)
        #     s.insert(ind, '.')
        #     return "".join(s)
        #
        # def redo_insert(s, ind):
        #     """ 回溯 撤销s中刚才插入的'.'
        #         返回 撤销指定'.'的字符串s """
        #     s = list(s)
        #     _ = s.pop(ind)
        #     return "".join(s)
        #
        # def backtracking(s, startInd, pointNum):
        #     """ s在回溯递归中可能会插入'.'，所以作为参数
        #         pointNum作为递归终止的判定条件，当分割为4段/已插入3个'.'了 """
        #     # 回溯递归终止条件
        #     if pointNum == 3:
        #         if isVal(s.split('.')[-1]):
        #             res.append(s[:])
        #         return
        #     # 遍历
        #     for i in range(startInd, len(s)):
        #         if isVal(s[startInd:i + 1]):
        #             s = insert(s, i + 1)        # 可理解为：在原s中插入'.'
        #             pointNum += 1
        #         else:
        #             continue
        #         backtracking(s, i + 2, pointNum)
        #         s = redo_insert(s, i + 1)       # 刚才插入'.'的索引为 i+1
        #         pointNum -= 1
        #
        # res = []
        # backtracking(s, 0, 0)
        # return res

        # 使用path保存某种分割结果，不用考虑插入删除'.'
        def isVal(s):
            """ s是否合法 """
            if len(s) == 0 or (len(s) > 1 and s[0] == '0') or not 0 <= int(s) <= 255:
                return False
            return True

        def backtracking(startInd):
            # 回溯递归终止条件
            if startInd == len(s) and len(path) == 4:
                res.append(".".join(path))
                return
            # 遍历 不同的分割方式
            for i in range(startInd, len(s)):
                if isVal(s[startInd:i + 1]):
                    path.append(s[startInd:i + 1][:])
                    backtracking(i + 1)
                    path.pop()

        res = []
        path = []       # 保存某种分割结果
        backtracking(0)
        return res

    def subsets(self, nums: List[int]) -> List[List[int]]:
        """ 78.子集 """
        def backtracking(startInd):
            res.append(path[:])
            if startInd >= len(nums):
                return
            for i in range(startInd, len(nums)):
                path.append(nums[i])
                backtracking(i + 1)
                path.pop()

        res = []
        path = []
        backtracking(0)
        return res

    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        """ 90.子集II """
        def backtracking(startInd):
            res.append(path[:])
            if startInd >= len(nums):
                return
            for i in range(startInd, len(nums)):
                if i > 0 and nums[i - 1] == nums[i] and not used[i - 1]:
                    continue
                path.append(nums[i])
                used[i] = True
                backtracking(i + 1)
                path.pop()
                used[i] = False

        res = []
        path = []
        used = [False] * (len(nums))
        nums.sort()
        backtracking(0)
        return res

    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        """ 491.递增子序列 """
        def backtracking(startInd):
            used = []
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
                path.pop()

        res = []
        path = []
        backtracking(0)
        return res

    def permute(self, nums: List[int]) -> List[List[int]]:
        """ 46.全排列 """
        def backtracking():
            if len(path) == len(nums):
                res.append(path[:])
                return
            for i in range(0, len(nums)):
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

if __name__ == "__main__":
    sl = Solution()

    nums = [1,2,3]
    print(sl.permute(nums))
