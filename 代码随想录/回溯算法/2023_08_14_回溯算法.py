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
        # https://programmercarl.com/0216.%E7%BB%84%E5%90%88%E6%80%BB%E5%92%8CIII.html#%E6%80%9D%E8%B7%AF


if __name__ == '__main__':
    sl = Solution()

    print(sl.combine(4, 2))