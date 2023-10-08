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


if __name__ == "__main__":
    sl = Solution()

    n = 4
    k = 2
    print(sl.combine(n, k))

    """
    回溯三部曲：
        1.函数参数
        2.终止条件
        3.单层搜索
    """