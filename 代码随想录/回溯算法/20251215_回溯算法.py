from typing import List


class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        """ 77.组合 """
        # res = []
        # path = []

        # def backtrack(startInd):
        #     if len(path) == k:
        #         res.append(path[:])
        #         return
        #     for i in range(startInd, n + 1):
        #         path.append(i)
        #         backtrack(i + 1)
        #         path.pop()
        
        # backtrack(1)
        # return res

        ## 剪枝优化
        res = []
        path = []

        def backtrack(startInd):
            """ startInd用于下层递归的起始位置 """
            if len(path) == k:
                res.append(path[:])
                return
            for i in range(startInd, n - (k - len(path)) + 1 + 1):
                path.append(i)
                backtrack(i + 1)
                path.pop()
        
        backtrack(1)
        return res

    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        """ 216.组合总和III """
        # res = []
        # path = []

        # def backtrack(startInd):
        #     if len(path) == k:
        #         if sum(path) == n:
        #             res.append(path[:])
        #         return
        #     for i in range(startInd, 10):
        #         path.append(i)
        #         backtrack(i + 1)
        #         path.pop()
        
        # backtrack(1)
        # return res

        ## 优化剪枝 根据n
        # res = []
        # path = []

        # def backtrack(startInd):
        #     # 剪枝
        #     if sum(path) > n:
        #         return
        #     # 终止条件
        #     if len(path) == k:
        #         if sum(path) == n:
        #             res.append(path[:])
        #         return
        #     # 递归逻辑
        #     for i in range(startInd, 10):
        #         path.append(i)
        #         backtrack(i + 1)
        #         path.pop()
        
        # backtrack(1)
        # return res

        ## 优化剪枝 根据k
        res = []
        path = []

        def backtrack(startInd):
            if len(path) == k:
                if sum(path) == n:
                    res.append(path[:])
                return
            for i in range(startInd, 9 - (k - len(path)) + 1 + 1):
                path.append(i)
                backtrack(i + 1)
                path.pop()
        
        backtrack(1)
        return res
    
    def letterCombinations(self, digits: str) -> List[str]:
        """ 17.电话号码的字母组合 """
        d2l = dict(zip(
            list('23456789'),
            ['abc', 'def', 'ghi', 'jkl', 'mno', 'pqrs', 'tuv', 'wxyz']
        ))
        
        res = []
        path = ''

        def backtrack(startInd):
            """ 注意: 多个集合求组合, startInd的作用是标识正处理字符的下标 """
            nonlocal path
            if len(path) == len(digits):
                res.append(path)
            for i in range(startInd, len(digits)):
                letters = d2l[digits[i]]
                for l in letters:
                    path += l
                    backtrack(i + 1)
                    path = path[:-1]
        
        backtrack(0)
        return res
    
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        """ 39.组合总和 """
        ## 元素可重复用 -> startInd不加1
        # res = []
        # path = []

        # def backtrack(startInd):
        #     if sum(path) >= target:
        #         if sum(path) == target:
        #             res.append(path[:])
        #         return
        #     for i in range(startInd, len(candidates)):
        #         path.append(candidates[i])
        #         backtrack(i)
        #         path.pop()

        # backtrack(0)
        # return res

        ## 剪枝优化
        res = []
        path = []
        candidates.sort()

        def backtrack(startInd):
            if sum(path) == target:
                res.append(path[:])
                return
            for i in range(startInd, len(candidates)):
                if sum(path) + candidates[i] > target:      # 剪枝
                    return
                path.append(candidates[i])
                backtrack(i)
                path.pop()
        
        backtrack(0)
        return res
    
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        """ 40.组合总和II """
        ## 元素有重复, 不能重复用 -> 去重
        res = []
        path = []
        candidates.sort()
        used = [False] * len(candidates)

        def backtrack(startInd):
            if sum(path) >= target:
                if sum(path) == target:
                    res.append(path[:])
                return
            for i in range(startInd, len(candidates)):
                if i > 0 and candidates[i - 1] == candidates[i] and used[i - 1] == False:
                    continue
                path.append(candidates[i])
                used[i] = True
                backtrack(i + 1)
                used[i] = False
                path.pop()
        
        backtrack(0)
        return res
    
    def partition(self, s: str) -> List[List[str]]:
        """ 131.分割回文串 """
        ## 分割问题--回溯
        ## (一个)组合问题 -> startInd
        ## 判断回文逻辑 -> startInd作为起点
        def isValid(s):
            l, r = 0, len(s) - 1
            while l <= r:
                if s[l] != s[r]:
                    return False
                l += 1
                r -= 1
            return True
        
        res = []
        path = []

        def backtrack(startInd):
            if startInd == len(s):
                res.append(path[:])
                return
            for i in range(startInd, len(s)):
                if isValid(s[startInd:i + 1]):
                    path.append(s[startInd:i + 1])
                    backtrack(i + 1)
                    path.pop()
        
        backtrack(0)
        return res

if __name__ == '__main__':
    """
    回溯题型:
        组合
        分割
        子集
        排列
        棋盘问题[optional]
    回溯/递归三部曲:
        1.回溯函数参数和返回值
        2.终止条件
        3.回溯搜索遍历过程
    """
    sl = Solution()

    s = "aab"
    print(sl.partition(s))