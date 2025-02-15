from typing import List

class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        """ 77.组合 """
        # 回溯三部曲:
        #   1.递归函数的参数和返回值
        #   2.回溯终止条件
        #   3.单层搜索过程
        # 时间:O(C(n * k)) 空间:O(C(n * k) * k)
        # res = []
        # path = []
        # def backtracking(n, k, startInd):
        #     if len(path) == k:
        #         res.append(path[:])         # res.append(path) 则res中每次保存的是path的引用, 而path在回溯过程中是变化的, 最终res中保存的所有元素都指向同一地址.
        #         return
        #     for i in range(startInd, n + 1):
        #         path.append(i)
        #         backtracking(n, k, i + 1)
        #         path.pop()
        # backtracking(n, k, 1)
        # return res

        # 剪枝
        path = []
        res = []
        def backtracking(n, k, start_ind):
            if len(path) == k:
                res.append(path[:])
                return
            for i in range(start_ind, n - (k - len(path)) + 1 + 1):
                path.append(i)
                backtracking(n, k, i + 1)
                path.pop()
        backtracking(n,k,1)
        return res
    
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        """ 216.组合总和III """
        # path = []
        # res = []
        # def backtracking(k, n, start_ind):
        #     if sum(path) == n and len(path) == k:
        #         res.append(path[:])
        #         return
        #     for i in range(start_ind, 9 + 1):
        #         path.append(i)
        #         backtracking(k, n, i + 1)
        #         path.pop()
        # backtracking(k, n, 1)
        # return res

        # 剪枝
        # 其实只需改动for
        path = []
        res = []
        def backtracking(k, n, start_ind):
            if len(path) == k:
                if sum(path) == n:
                    res.append(path[:])
                return
            for i in range(start_ind, 9 - (k - len(path)) + 1 + 1):     # 两个+1: 第一个+1表示最晚的起始索引, 第二个+1是python语法需要
                path.append(i)
                backtracking(k, n, i + 1)
                path.pop()
        backtracking(k, n, 1)
        return res
    
    def letterCombinations(self, digits: str) -> List[str]:
        """ 17.电话号码的字母组合 \n
            77.组合 和 216.组合总和III 是求同一个集合的组合, 而本题是求不同集合的组合--区别是start_ind的作用 """
        digit2letter = {
            2: "abc", 
            3: "def",
            4: "ghi",
            5: "jkl",
            6: "mno",
            7: "pqrs",
            8: "tuv",
            9: "wxyz"
        }
        s = ""
        res = []
        def backtracking(digits, ind):
            nonlocal s
            if ind == len(digits):
                res.append(s)
                return
            digit = int(digits[ind])
            letters = digit2letter[digit]
            for i in range(len(letters)):                
                s += letters[i]
                backtracking(digits, ind + 1)
                s = s[:-1]
        if len(digits) == 0:
            return res
        backtracking(digits, 0)
        return res

if __name__ == "__main__":
    sl = Solution()

    digits = "2"
    print(sl.letterCombinations(digits))
"""
回溯三部曲:
    1.递归函数的参数和返回值
    2.回溯终止条件
    3.单层搜索过程
"""