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

if __name__ == '__main__':
    """
    回溯三部曲:
        1.确定递归函数参数和返回值
        2.回溯函数终止条件
        3.单层搜索逻辑
    """
    sl = Solution()
    sl.letterCombinations('')