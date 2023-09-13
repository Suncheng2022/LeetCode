from typing import List


class Solution:
    def fib(self, n: int) -> int:
        # dp = [0] * (n + 1)     # dp[i] 索引第i个斐波那契数是dp[i]
        # if n == 0:      # 根据题意，记得特殊处理n=0
        #     return dp[-1]
        # dp[0] = 0
        # dp[1] = 1
        # for i in range(2, n + 1):
        #     dp[i] = dp[i - 1] + dp[i - 2]
        # return dp[-1]

        # 优化下空间复杂度
        if n == 0:
            return 0
        dp = [0, 1]     # 就用dp长度为2
        for i in range(2, n + 1):
            f = sum(dp)
            dp[0] = dp[1]
            dp[1] = f
        return dp[-1]

    def climbStairs(self, n: int) -> int:
        """ 70.爬楼梯 """
        if n == 1:      # 特殊处理n=1的情况
            return 1
        dp = [0] * (n + 1)      # dp[i] 表示爬上第i阶楼梯有dp[i]种方法；《代码随想录》不考虑dp[0] 因为没意义，题目已说明n>=1为正整数
        dp[1] = 1
        dp[2] = 2
        for i in range(3, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[-1]

    def minCostClimbingStairs(self, cost: List[int]) -> int:
        """ 746.使用最小花费爬楼梯
            注意题目描述，从索引0或1起跳，到达索引0或1不花钱，从0或1跳就要花钱了 """
        n = len(cost)
        dp = [0] * (n + 1)      # dp[i] 到达索引0台阶所需最小花费; dp[0] dp[1]均初始化为0
        for i in range(2, n + 1):
            dp[i] = min(dp[i - 1] + cost[i - 1],    # 从索引i-1台阶跳到索引i台阶——到达索引i-1台阶最小花费dp[i-1]，起跳花费cost[i-1]
                        dp[i - 2] + cost[i - 2])    # 同上
        return dp[-1]

    def uniquePaths(self, m: int, n: int) -> int:
        """ 62.不同路径 """
        dp = [[1] * n] + [[1] + [0] * (n - 1) for _ in range(m - 1)]    # dp[i][j] 从坐标[0,0]走到[i,j]的路径数
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[-1][-1]

    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        """ 63.不同路径II """
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        dp = [[0] * n for _ in range(m)]    # dp[i][j] 从位置[0,0]走到位置[i,j] 的 不同路径数
        # 初始化dp 首行首列遇到障碍物后面的都会是0
        for i in range(m):
            if obstacleGrid[i][0] == 1:
                break
            dp[i][0] = 1
        for j in range(n):
            if obstacleGrid[0][j] == 1:
                break
            dp[0][j] = 1
        for i in range(1, m):       # 注意两个for循环起始索引
            for j in range(1, n):
                if obstacleGrid[i][j] == 1:
                    continue
                dp[i][j] = dp[i][j - 1] + dp[i - 1][j]
        return dp[-1][-1]

    def integerBreak(self, n: int) -> int:
        """ 343.整数拆分
            没思路：dp[i] 将数字i拆分最大乘积
            """
        dp = [0] * (n + 1)      # 要拆分到n，所以要有n+1个数
        dp[2] = 1       # dp[0] dp[1]不符合dp的设计要求，忽略不使用
        for i in range(3, n + 1):
            for j in range(1, i // 2 + 1):       # 从i上拆分下一个j来, 同时注意其范围
                dp[i] = max(dp[i], j * (i - j), j * dp[i - j])      # j的for循环会多次计算dp[i]，取最大的dp[i]而已
        return dp[-1]

    def numTrees(self, n: int) -> int:
        """ 96.不同的二叉搜索树
            没思路：dp[i] i个节点组成的BST数量+=dp[j]*dp[i-1-j] 即遍历所有左右子树的情况 """
        dp = [0] * (n + 1)      # 根据dp定义，最终答案是dp[n] 所以元素一共n+1个
        dp[0] = 1       # 初始化，因为递推公式的需要
        for i in range(1, n + 1):   # 计算dp[i]
            for j in range(i):      # 其中一棵子树所能拥有的节点数量为 0~i-1，-1表示父节点占用1个
                dp[i] += dp[j] * dp[i - 1 - j]      # 与《代码随想录稍有区别》--只要两棵子树节点数量相加为i-1即可
        return dp[-1]


if __name__ == '__main__':
    sl = Solution()

    n = 1
    print(sl.numTrees(n))
