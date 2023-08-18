""" https://programmercarl.com/%E5%9B%9E%E6%BA%AF%E7%AE%97%E6%B3%95%E7%90%86%E8%AE%BA%E5%9F%BA%E7%A1%80.html """
from typing import List


class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        """ 77.组合
            中等 https://programmercarl.com/0077.%E7%BB%84%E5%90%88.html#%E7%AE%97%E6%B3%95%E5%85%AC%E5%BC%80%E8%AF%BE """
        # res = []
        #
        # def backtracking(startIndex, path):
        #     if len(path) == k:
        #         res.append(path[:])
        #         return
        #     # for i in range(startIndex, n + 1):      # 剪枝优化 前
        #     for i in range(startIndex, n - (k - len(path)) + 1 + 1):        # 剪枝优化 后；还需要k-len(path)个元素，则最大起始索引为n-k-len(path)+1，python遍历还需要再+1
        #         path.append(i)
        #         backtracking(i + 1, path)
        #         path.pop()
        #
        # backtracking(1, [])
        # return res

        # 再写一遍
        def backtracking(startInd):
            # 如果遍历到叶节点，收集结果、结束本次回溯
            if len(path) == k:
                res.append(path[:])
                return
            for i in range(startInd, n - (k - len(path)) + 1 + 1):      # 最晚从 索引n - (k - len(path)) + 1开始遍历才能收集到k个元素，否则收集不到；最后的+1是Python语法
                path.append(i)
                backtracking(i + 1)     # 注意参数，不是startInd+1
                path.pop()

        res = []
        path = []       # 不必担心for每第一次调用回溯要空的path，因为递归后要删除上一个结果
        backtracking(1)
        return res

    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        """ 216.组合总和III """
        # def backtracking(startInd):
        #     # 回溯递归 的 终止条件，结束本次回溯
        #     if sum(path) > n:       # 剪枝操作
        #         return
        #     if len(path) == k:
        #         if sum(path) == n:
        #             res.append(path[:])
        #         return
        #     for i in range(startInd, 9 + 1):    # 从1~9中选数; 也是起始索引
        #         path.append(i)
        #         backtracking(i + 1)
        #         path.pop()
        #
        # res = []
        # path = []
        # backtracking(1)
        # return res

        # 剪枝 1.放在回溯函数的开始 【更好理解】
        # 剪枝 2.放在for循环
        def backtracking(startInd):
            # 回溯递归 的 结束条件
            if len(path) == k:
                if sum(path) == n:
                    res.append(path[:])
                return
            # 遍历
            for i in range(startInd, 9 - (k - len(path)) + 1 + 1):      # 剪枝 索引最大的 起始点；也是遍历范围
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
        digit2letter = dict(zip(range(2, 10), ['abc', 'def',
                                               'ghi', 'jkl', 'mno',
                                               'pqrs', 'tuv', 'wxyz']))
        # res = []
        #
        # def backtracking(index, curPath):
        #     # 递归终止条件
        #     if index == len(digits):
        #         res.append(curPath)
        #         return
        #     # 遍历
        #     letters = digit2letter[ord(digits[index]) - ord('0')]
        #     for c in letters:
        #         curPath += c
        #         backtracking(index + 1, curPath)
        #         curPath = curPath[:-1]      # 这里要注意 用字符串转还错了
        #
        # backtracking(0, '')
        # return res

        # 再写一遍
        def backtracking(index, path):
            """ index 当前遍历字符索引; path 当前字符组合 """
            # 回溯终止条件: 遍历到最后一个字符，收集结果，结束本次回溯
            if index == len(digits):
                res.append(path)
                return
            # 遍历，注意 这里遍历的是 “每个按键的所有字符”
            letters = digit2letter[ord(digits[index]) - ord('0')]
            for c in letters:
                path += c
                backtracking(index + 1, path)
                path = path[:-1]

        res = []
        path = ""
        backtracking(0, path)
        return res

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        """ 39.组合总和 """
        def backtracking(startInd):
            # 回溯递归的终止条件
            if sum(path) > target:
                return
            elif sum(path) == target:
                res.append(path[:])
                return
            # 遍历
            for i in range(startInd, len(candidates)):
                # 剪枝操作[本题剪枝操作，首先要把数组排序]
                if sum(path) + candidates[i] > target:
                    break
                path.append(candidates[i])
                backtracking(i)     # 递归调用的时候startInd不+1 表示可重复选取
                path.pop()      # 回溯

        candidates.sort()       # 本题剪枝操作，要把数组排序
        res = []
        path = []
        backtracking(0)
        return res

    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        """ 40.组合总和II """
        def backtracking(startInd):
            if sum(path) == target:
                res.append(path[:])
                return
            for i in range(startInd, len(candidates)):
                # 剪枝，否则超时
                if sum(path) + candidates[i] > target:
                    continue
                if i > 0 and candidates[i - 1] == candidates[i] and used[i - 1] == False:   # 同层重复的元素，跳过
                    continue
                path.append(candidates[i])
                used[i] = True
                backtracking(i + 1)
                used[i] = False
                path.pop()

        res = []
        path = []
        used = [False] * len(candidates)
        candidates.sort()       # 切记，本题这种题目需要排序 达到去重的目的
        backtracking(0)
        return res

    def partition(self, s: str) -> List[List[str]]:
        """ 131.分割回文串 """
        def isPal(s):
            l, r = 0, len(s) - 1
            while l <= r:
                if s[l] != s[r]:
                    return False
                l += 1
                r -= 1
            return True

        def backtracking(startIndex):
            # 回溯递归的终止条件
            if startIndex >= len(s):
                res.append(path[:])
                return
            # 遍历，横向遍历回溯的二叉树示例图
            for i in range(startIndex, len(s)):
                # 这里要注意索引s[startIndex:i + 1]  要+1
                if isPal(s[startIndex:i + 1]):
                    path.append(s[startIndex:i + 1][:])     # 终于想明白了，是横向遍历的时候 考虑不同的分支(比如第一层子节点)
                else:
                    continue

                backtracking(i + 1)
                path.pop()

        res = []
        path = []
        backtracking(0)
        return res

if __name__ == '__main__':
    sl = Solution()

    s = 'a'
    print(sl.partition(s))