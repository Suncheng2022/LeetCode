from typing import List


class Solution:
    def fib(self, n: int) -> int:
        """ 509.斐波那契数 """
        if n <= 1:
            return n
        dp = [0] * (n + 1)
        dp[0], dp[1] = 0, 1
        for i in range(2, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[n]

    def climbStairs(self, n: int) -> int:
        """ 70.爬楼梯 """
        if n <= 1:
            return n
        dp = [0] * (n + 1)
        dp[1], dp[2] = 1, 2     # 爬到索引i楼梯，有dp[i]种方法; 不使用dp[0]，因为dp[0]如何初始化原因解释不通
        for i in range(3, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[n]

        # 优化一下空间复杂度
        # if n <= 1:
        #     return n
        # dp = [1, 2]
        # for i in range(3, n + 1):
        #     sum = dp[0] + dp[1]
        #     dp[0] = dp[1]
        #     dp[1] = sum
        # return dp[1]

    def minCostClimbingStairs(self, cost: List[int]) -> int:
        """ 746.使用最小花费爬楼梯 """
        dp = [0] * (len(cost) + 1)      # dp的含义要明确，到达下标i台阶所需要的最小花费为dp[i]
        dp[0], dp[1] = 0, 0     # 题目明确说了，可以选择从下标为0或1的台阶开始爬，即到达下标0或1的台阶不花费，继续往上爬就要花费体力
        for i in range(2, len(cost) + 1):
            # 状态转移公式：dp[i-1]转移到dp[i]——即从下标i-1继续往上爬就需要花费体力了，体力就是cost[i-1]；下标i-2往上爬类似
            dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])
        return dp[-1]

        # 优化 空间复杂度
        # dp = [0] * 2
        # dp[0], dp[1] = 0, 0     # 起点是下标0或1，到达起点不花费，继续往上爬才需要花费体力
        # for i in range(2, len(cost) + 1):
        #     dp_i = min(dp[0] + cost[i - 2], dp[1] + cost[i - 1])    # 有点绕，想一下索引i=2如何得到吧
        #     dp[0] = dp[1]
        #     dp[1] = dp_i
        # return dp[1]

    def uniquePaths(self, m: int, n: int) -> int:
        """ 62.不同路径 """
        dp = [[1] * n] + [[1] + [0] * (n - 1) for _ in range(m - 1)]    # 从位置[0,0]到达[i,j]的不同路径数
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[-1][-1]

    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        """ 63.不同路径II """
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        dp = [[0] * n for _ in range(m)]
        # 初始化dp首行
        for i in range(n):
            if obstacleGrid[0][i] == 1:
                break
            dp[0][i] = 1
        # 初始化dp首列
        for i in range(m):
            if obstacleGrid[i][0] == 1:
                break
            dp[i][0] = 1
        # print(dp)
        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] == 0:
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[-1][-1]

    def integerBreak(self, n: int) -> int:
        """ 343.整数拆分 """
        # 重写一遍
        dp = [0] * (n + 1)      # 拆分i得到的最大乘积为dp[i]
        dp[2] = 1       # 初始化dp，不使用dp[0]、dp[1] 因为不符合dp定义，直接从dp[2]开始计算
        for i in range(3, n + 1):
            for j in range(1, i):       # 其实j可以拆分到 n/2,因为拆分成近似大小的数字乘积才最大，拆分成n/2也就是拆分成两个数，两个数差不多都是n/2的情况
                dp[i] = max(dp[i], (i - j) * j, dp[i - j] * j)      # dp递推公式
        return dp[-1]

        # dp = [0] * (n + 1)
        # dp[2] = 1       # dp计算过程忽略dp[0]、dp[1]，因为这两者不符合dp含义；所以从dp[2]开始
        # for i in range(3, n + 1):
        #     for j in range(1, i):
        #         dp[i] = max(dp[i], (i - j) * j, dp[i - j] * j)      # dp[i]会通过for j循环计算多次，保留最大的而已
        # return dp[-1]

    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        """ 63.不同路径II """
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        dp = [[0] * n for _ in range(m)]
        # 初始化dp首行
        for i in range(n):
            if obstacleGrid[0][i] == 1:
                break
            dp[0][i] = 1
        # 初始化dp首列
        for i in range(m):
            if obstacleGrid[i][0] == 1:
                break
            dp[i][0] = 1
        # print(dp)
        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] == 0:
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[-1][-1]

    def integerBreak(self, n: int) -> int:
        """ 343.整数拆分 """
        # 重写一遍
        dp = [0] * (n + 1)      # 拆分i得到的最大乘积为dp[i]
        dp[2] = 1       # 初始化dp，不使用dp[0]、dp[1] 因为不符合dp定义，直接从dp[2]开始计算
        for i in range(3, n + 1):
            for j in range(1, i):       # 其实j可以拆分到 n/2,因为拆分成近似大小的数字乘积才最大，拆分成n/2也就是拆分成两个数，两个数差不多都是n/2的情况
                dp[i] = max(dp[i], (i - j) * j, dp[i - j] * j)      # dp递推公式
        return dp[-1]

        # dp = [0] * (n + 1)
        # dp[2] = 1       # dp计算过程忽略dp[0]、dp[1]，因为这两者不符合dp含义；所以从dp[2]开始
        # for i in range(3, n + 1):
        #     for j in range(1, i):
        #         dp[i] = max(dp[i], (i - j) * j, dp[i - j] * j)      # dp[i]会通过for j循环计算多次，保留最大的而已
        # return dp[-1]

    def numTrees(self, n: int) -> int:
        """ 96.不同的二叉搜索树 """
        dp = [0] * (n + 1)      # 1~i组成的节点，能组成BST的数量
        dp[0] = 1       # n为0的时候只有一种可能——空树
        for i in range(1, n + 1):
            for j in range(0, i):   # 其中一棵子树由j个节点组成 0<=j<=i-1, 因为根节点需要耗费1个节点
                dp[i] += dp[j] * dp[i - 1 - j]
        return dp[-1]

    def test_2_wei_bag_problem1(self):
        """ 01背包问题 导读1
            代码作参考而已 """
        weights = [1, 3, 4]
        values = [15, 20, 30]
        bagweight = 4

        # 初始化dp
        # 【注意】dp纵向表示可选择物品的索引0~i，注意0也表示物品，而不是不选；dp横向表示背包容量0~bagweight，注意有0
        # dp = [[0] * (bagweight + 1) for _ in range(len(weights))]
        # for j in range(1, len(dp[0])):
        #     if j >= weights[0]:
        #         dp[0][j] = values[0]
        # print(dp)
        #
        # # 遍历：先遍历物品 或 先遍历背包 都可以，看哪个好理解
        # for i in range(1, len(weights)):
        #     for j in range(1, bagweight + 1):
        #         # 物品i 放进去 or 没放进去，哪个价值更大
        #         dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weights[i]] + values[i])
        # print(dp[-1][-1])

        # 重写一遍
        # dp[i][j] 从0~i物品中选，背包大小为j，所能装下的最大价值
        dp = [[0] * (bagweight + 1) for _ in range(len(weights))]
        # 初始化 要同时考虑递推公式，思考需要先初始化哪些-->通常是由大约左上角元素推当前元素
        # j=0时肯定dp=0；i=0时dp首行能放下weight[0]的位置都初始化为weight[0]
        for j in range(weights[0], bagweight + 1):
            dp[0][j] = values[0]

        # 遍历
        for i in range(1, len(weights)):
            for j in range(1, bagweight + 1):
                if j < weights[i]:      # 放不下物品i
                    dp[i][j] = dp[i - 1][j]
                else:                   # 放得下物品i
                    dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weights[i]] + values[i])
        print(dp[-1][-1])

    def test_1_wei_bag_problem(self):
        """ 01背包问题 导读2
            代码仅作参考 """
        weights = [1, 3, 4]
        values = [15, 20, 30]
        bagweight = 4

        # dp = [0] * (bagweight + 1)
        # for i in range(len(weights)):
        #     for j in range(bagweight, weights[i] - 1, -1):
        #         dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
        # return dp[-1]
        
        # 重写一遍
        # 一维dp[j]  表示背包容量j时，能装下的最大价值
        # 初始化 dp[0]=0  都放不下任何物品，价值自然是0
        dp = [0] * (bagweight + 1)
        
        for i in range(len(weights)):
            for j in range(bagweight, weights[i] - 1, -1):      # 隐含说明了 j是能放下weight[i]的
                dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
        return dp[-1]

        
if __name__ == '__main__':
    sl = Solution()

    print(sl.test_1_wei_bag_problem())

    """
    动态规划五部曲：
        1.确定dp数组以及下标的含义
        2.确定递推公式
        3.dp数组如何初始化
        4.确定遍历顺序
        5.举例推导dp数组
    """