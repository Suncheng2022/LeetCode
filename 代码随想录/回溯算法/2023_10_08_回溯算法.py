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


if __name__ == "__main__":
    sl = Solution()

    digits = ""
    print(sl.letterCombinations(digits))

    """
    回溯三部曲：
        1.函数参数
        2.终止条件
        3.单层搜索
    """