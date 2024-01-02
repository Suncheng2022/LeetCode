from typing import List


class Solution:
    def fib(self, n: int) -> int:
        """ 509.斐波那契数 """
        # dp = [0] * (n + 1)
        # if n < 1:                  # 处理边界
        #     return dp[n]
        # dp[0] = 0
        # dp[1] = 1
        # for i in range(2, n + 1):
        #     dp[i] = dp[i - 1] + dp[i - 2]
        # return dp[-1]

        # 优化一下空间复杂度
        if n <= 1:                  # 处理边界
            return n
        dp = [0] * 2                # dp初始化长度为2
        dp[0] = 0
        dp[1] = 1
        for i in range(2, n + 1):
            tmp = sum(dp)
            dp[0] = dp[1]
            dp[1] = tmp
        return dp[-1]

    def climbStairs(self, n: int) -> int:
        """ 70.爬楼梯
            递推公式 dp[i] = dp[i - 1] + dp[i - 2] """
        # if n == 1:
        #     return 1
        # dp = [0] * (n + 1)          # dp[i] 爬到第i阶楼梯有dp[i]种方法
        # dp[1] = 1                   # 初始化，不考虑dp[0]了，没实际意义，题目也已说明n>=1
        # dp[2] = 2
        # for i in range(3, n + 1):
        #     dp[i] = dp[i - 1] + dp[i - 2]
        # return dp[-1]

        # 优化空间复杂度
        # if n == 1:
        #     return 1
        # dp = [0] * 2                # 优化空间
        # dp[0] = 1
        # dp[1] = 1
        # for i in range(2, n + 1):
        #     tmp = sum(dp)
        #     dp[0] = dp[1]
        #     dp[1] = tmp
        # return dp[-1]

        # 或者可以这样优化空间
        if n == 1:
            return 1
        dp = [0] * 3
        dp[1] = 1
        dp[2] = 2
        for i in range(3, n + 1):
            tmp = dp[1] + dp[2]
            dp[1] = dp[2]
            dp[2] = tmp
        return dp[-1]

    def minCostClimbingStairs(self, cost: List[int]) -> int:
        """ 746.使用最小花费爬楼梯 """
        # 虽然自己写对了，还是看了一下《代码随想录》
        # n = len(cost)               # 一共多少阶楼梯
        # dp = [0] * (n + 1)          # dp[i] 爬到索引第i阶楼梯所需最小花费；得能爬到第n阶楼梯呀，所以索引要到n
        # dp[0] = 0                   # 初始化dp[0] dp[1]
        # dp[1] = 0
        # for i in range(2, n + 1):
        #     dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])       # 递推公式，到达第i阶 可以从第i-1阶、第i-2阶跳
        # return dp[-1]

        # 优化空间，虽然个人不太推荐，还是多会一个吧
        n = len(cost)
        dp = [0] * 2
        dp[0] = 0
        dp[1] = 0
        for i in range(2, n + 1):
            tmp = min(dp[1] + cost[i - 1], dp[0] + cost[i - 2])                   # 注意，dp长度为2了
            dp[0] = dp[1]
            dp[1] = tmp
        return dp[-1]

    def uniquePaths(self, m: int, n: int) -> int:
        """ 62.不同路径 """
        # dp = [[0] * n for _ in range(m)]            # dp[i][j] 到达坐标[i,j]有dp[i][j]种方法
        # for j in range(n):                          # 初始化 dp首行、首列为1
        #     dp[0][j] = 1
        # for i in range(m):
        #     dp[i][0] = 1
        # for i in range(1, m):
        #     for j in range(1, n):
        #         dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        # return dp[-1][-1]

        # 《代码随想录》稍微优化空间，还是建议上面写的，明了呀
        dp = [1] * n                        # 只使用1列，初始化为1
        for i in range(1, m):
            for j in range(1, n):
                dp[j] += dp[j - 1]          # 其实还是 当前=上+左
        return dp[-1]

    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        """ 63.不同路径II """
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        dp = [[0] * n for _ in range(m)]            # dp[i][j] 到达位置[i,j]有多少种方法
        for i in range(m):
            if obstacleGrid[i][0] == 1:             # 初始化注意，遇到'障碍物'就停止
                break
            dp[i][0] = 1
        for j in range(n):
            if obstacleGrid[0][j] == 1:
                break
            dp[0][j] = 1
        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] == 1:
                    continue
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[-1][-1]

    def integerBreak(self, n: int) -> int:
        """ 343.整数拆分
            注意递推公式，内层for多次计算同一个dp[i]，取最大的 """
        # 内层for范围不好确定，建议下面一种方法
        # from math import sqrt
        #
        # dp = [0] * (n + 1)                          # dp[i] 拆分整数i可获得的最大乘积
        # dp[2] = 1                                   # dp[0] dp[1]根本没法拆分，没意义
        # for i in range(3, n + 1):
        #     for j in range(1, int(sqrt(i)) + 2):    # int()是舍弃小数部分，也即向下取整，不会四舍五入；最安全的还是 i // 2 + 1
        #         dp[i] = max(dp[i], j * (i - j), j * dp[i - j])     # 拆分为2个、或2个以上，哪种方式乘积大
        # return dp[-1]

        # 《代码随想录》
        dp = [0] * (n + 1)                              # dp[i] 拆分i能得到的最大乘积
        dp[2] = 1                                       # dp[0]、dp[1]根本没意义
        for i in range(3, n + 1):
            for j in range(1, i // 2 + 1):              # 可以知道，一个数拆分为m个近似的数乘积最大，m至少得是2吧，所以j的范围 i // 2，range取到则 + 1
                dp[i] = max(dp[i], j * (i - j), j * dp[i - j])
        return dp[-1]

    def numTrees(self, n: int) -> int:
        """ 96.不同的二叉搜索树 """
        dp = [0] * (n + 1)                              # dp[i] i个节点能组成dp[i]个BST
        dp[0] = 1                                       # 纯为递推公式
        dp[1] = 1
        for i in range(2, n + 1):
            for j in range(i):                          # 其中一棵子树有j个节点，另一棵子树有i-1-j个节点，留下1个节点作父节点
                dp[i] += dp[j] * dp[i - 1 - j]
        return dp[-1]

    def test_2_wei_bag_problem1(self):
        """ 01背包导读 二维数组 """
        weights = [1, 3, 4]
        values = [15, 20, 30]
        bagweight = 4
        dp = [[0] * (bagweight + 1) for _ in range(len(weights))]       # m行n列 m:物品个数 n:背包容量
        for j in range(weights[0], bagweight + 1):                      # 首列初始化为0，首行看能否放下weights[0]
            dp[0][j] = values[0]
        for i in range(1, len(weights)):                                # 先遍历物品、再遍历容量 更好理解
            for j in range(weights[i], bagweight + 1):
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weights[i]] + values[i])
        return dp[-1][-1]       # 返回35

    def test_1_wei_bag_problem(self):
        """ 01背包 一维数组 """
        weights = [1, 3, 4]
        values = [15, 20, 30]
        bagweight = 4
        dp = [0] * (bagweight + 1)                  # 一维数组，去掉物品i的维度，只保留容量的维度
        for i in range(len(weights)):               # 一维数组，必须先遍历物品、再遍历背包，且内层倒序
            for j in range(bagweight, weights[i] - 1, - 1):
                dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
        return dp[-1]       # 35

"""
动规五部曲：
    1.明确dp下标及dp[i]的含义
    2.确定递推公式
    3.初始化
    4.确定遍历顺序
    5.
"""
if __name__ == "__main__":
    sl = Solution()

    print(sl.test_1_wei_bag_problem())