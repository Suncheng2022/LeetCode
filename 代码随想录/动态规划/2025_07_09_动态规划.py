"""
07.09   是的, 该走了
"""
from typing import List

class Solution:
    def fib(self, n: int) -> int:
        """ 509.斐切那波数 """
        # 时间:O(n) 空间:O(1)
        # if n <= 1:
        #     return n
        # dp = [0, 1]     # 保留两个结果
        # for i in range(2, n + 1):
        #     tmp = sum(dp)
        #     dp[0], dp[1] = dp[1], tmp
        # return dp[-1]

        # 递归
        # 时间:O(2^n) 二叉树节点数 空间:O(n) 栈的最大深度就是最长调用链
        if n <= 1:
            return n
        return self.fib(n - 1) + self.fib(n - 2)

        # 时间:O(n) 空间:O(n) 
        # if n <= 1:
        #     return n
        # dp = [0] * (n + 1)
        # dp[0], dp[1] = 0, 1
        # for i in range(2, n + 1):
        #     dp[i] = dp[i - 1] + dp[i - 2]
        # return dp[n]

    def climbStairs(self, n: int) -> int:
        """ 70.爬楼梯 """
        # 时间:O(n) 空间:O(1)
        if n <= 2:
            return n
        dp = [0] * 3            # dp[0]无意义
        dp[1], dp[2] = 1, 2     # 初始化
        for i in range(3, n + 1):
            tmp = dp[1] + dp[2]
            dp[1], dp[2] = dp[2], tmp
        return dp[-1]

        # 时间:O(n) 空间:O(n)
        # if n <= 2:
        #     return n
        # dp = [0] * (n + 1)      # dp[i]表示爬到第i个台阶有dp[i]种方法
        # dp[1] = 1
        # dp[2] = 2
        # for i in range(3, n + 1):
        #     dp[i] = dp[i - 1] + dp[i - 2]
        # return dp[-1]

    def minCostClimbingStairs(self, cost: List[int]) -> int:
        """ 746.使用最小花费爬楼梯 \n
            理解好题意, 可以从台阶0或1开始, 就是到达台阶0或1是不花钱的\n
                        最终的目的是越过所有台阶, 所以dp长度为n+1 \n
            可以参考下<代码随想录>本题推导步骤的图片 """
        # 时间:O(n) 空间:O(1)
        n = len(cost)
        dp = [0, 0]     # 优化空间, dp[i]通过前两项推导得来, 所以保留最新的两项即可
        for i in range(2, n + 1):
            _min = min(dp[-1] + cost[i - 1], dp[-2] + cost[i - 2])
            dp[0], dp[1] = dp[1], _min
        return dp[-1]

        # 时间:O(n) 空间:O(n)
        # n = len(cost)
        # if n <= 2:
        #     return min(cost)              # 最终目的是越过所有台阶, 即便只有2个台阶, 也要越呀
        # dp = [0] * (n + 1)                # dp[i]表示到达下标为i的台阶所需最小花费
        # for i in range(2, n + 1):
        #     dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])
        # return dp[-1]

    def uniquePaths(self, m: int, n: int) -> int:
        """ 62.不同路径 """
        # 时间:O(m*n) 空间:O(m*n)
        # if m == n == 1:
        #     return 1
        # dp = [[0] * n for _ in range(m)]    # dp[i][j]表示到达第i行第j列有dp[i][j]个不同路径
        # for i in range(m):
        #     dp[i][0] = 1
        # for j in range(n):
        #     dp[0][j] = 1
        # for i in range(1, m):
        #     for j in range(1, n):
        #         dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        # return dp[-1][-1]
        
        # 时间:O(m*n) 空间:O(n)
        dp = [1] * n        # 滚动数组
        for i in range(1, m):
            for j in range(1, n):
                dp[j] += dp[j - 1]
        return dp[-1]
    
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        """ 63.不同路径II """
        # 时间:O(m*n) 空间:O(m*n)
        # m, n = len(obstacleGrid), len(obstacleGrid[0])
        # dp = [[0] * n for _ in range(m)]        # dp[i][j]表示到达位置[i,j]有多少种不同方法
        # for i in range(m):
        #     if obstacleGrid[i][0] == 1:
        #         break
        #     dp[i][0] = 1
        # for j in range(n):
        #     if obstacleGrid[0][j] == 1:
        #         break
        #     dp[0][j] = 1
        # for i in range(1, m):
        #     for j in range(1, n):
        #         if obstacleGrid[i][j] == 1:
        #             continue
        #         dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        # return dp[-1][-1]

        # 时间:O(mxn) 空间:O(n)
        # 优化空间有点不好想, 主要是要处理遍历的时候
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        dp = [0] * n    # 滚动数组 优化空间
        for i in range(n):
            if obstacleGrid[0][i] == 1:
                break
            dp[i] = 1
        for i in range(1, m):
            for j in range(n):
                if obstacleGrid[i][j] == 1:
                    dp[j] = 0
                elif j > 0:     # 如果不加>0的限制条件, 会有dp[0] += dp[-1], 对不
                    dp[j] += dp[j - 1]
        return dp[-1]

    def integerBreak(self, n: int) -> int:
        """ 343.整数拆分 """
        # 时间:O(n^2) 空间:O(n)
        dp = [0] * (n + 1)        # dp[i]表示拆分i所得最大乘积
        dp[2] = 1                 # dp[0] dp[1]就没有意义
        for i in range(3, n + 1):
            for j in range(1, i // 2 + 1):
                dp[i] = max(dp[i], max((i - j) * j, dp[i - j] * j))     # 拆分公式/推导公式不好想出来
        return dp[-1]
    
    def numTrees(self, n: int) -> int:
        """ 96.不同的二叉搜索树 """
        # 时间:O(n^2) 空间:O(n)
        dp = [0] * (n + 1)      # dp[i]表示 i个不同节点/从1到i共i个不同节点 所组成不同二叉搜索树的数量
        dp[0] = 1
        for i in range(1, n + 1):
            for j in range(1, i + 1):
                dp[i] += dp[j - 1] * dp[i - j]      # 同样, 递推公式不好想
        return dp[-1]

if __name__ == '__main__':
    """
    动归五部曲: 
        1.确定dp数组及下标i含义
        2.确定递推公式
        3.dp数组初始化
        4.确定遍历顺序
        4.举例推导
    """
    sl = Solution()

    n = 1
    print(sl.numTrees(n))