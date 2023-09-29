from typing import List


class Solution:
    def fib(self, n: int) -> int:
        """ 509.斐波那契数
            简单 """
        # 时间：O(n) 空间：O(n)
        # if n <= 1:
        #     return n
        # dp = [0] * (n + 1)
        # dp[0] = 0
        # dp[1] = 1
        # for i in range(2, n + 1):
        #     dp[i] = dp[i - 1] + dp[i - 2]
        # return dp[-1]

        # 优化空间：O(1)
        if n <= 1:
            return n
        dp = [0, 1]
        for i in range(2, n + 1):
            tmp = sum(dp)
            dp[0] = dp[1]
            dp[1] = tmp
        return dp[-1]

    def climbStairs(self, n: int) -> int:
        """ 70.爬楼梯
            简单 """
        # 时间：O(n) 空间：O(n)
        # if n == 1:
        #     return 1
        # dp = [0] * (n + 1)      # dp[i] 爬到索引第i阶有dp[i]种方法; 不使用dp[0] 因为题目已规定n范围
        # dp[1] = 1
        # dp[2] = 2
        # for i in range(3, n + 1):
        #     dp[i] = dp[i - 1] + dp[i - 2]
        # return dp[-1]

        # 优化空间复杂度：O(1)
        if n == 1:
            return 1
        dp = [1, 2]
        for i in range(2, n):   # 注意for范围和上面不同，灵活运用吧
            tmp = sum(dp)
            dp[0] = dp[1]
            dp[1] = tmp
        return dp[-1]

    def minCostClimbingStairs(self, cost: List[int]) -> int:
        """ 746.使用最小花费爬楼梯
            简单 """
        # 时间：O(n) 空间：O(n)
        # n = len(cost)
        # dp = [0] * (n + 1)      # dp[i] 爬到索引第i阶楼梯最小花费；dp[-1]即越过最后一阶台阶到达楼顶；注意，越过最后一个台阶，才叫到达楼顶(不要再想自己把'平地'当台阶0了)
        # for i in range(2, n + 1):
        #     dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])
        # return dp[-1]

        # 优化时间复杂度 O(1)
        n = len(cost)
        dp = [0, 0]     # 因为dp由前2个状态推导来
        for i in range(2, n + 1):
            tmp = min(dp[0] + cost[i - 2], dp[1] + cost[i - 1])
            dp[0] = dp[1]
            dp[1] = tmp
        return dp[-1]

        # 我这么想不太对，虽然能过，但dp意义不准确
        # n = len(cost)
        # dp = [0] * (n + 1)      # dp[i] 到达索引i台阶的最小花费 【注意】索引0台阶意味着还没有上台阶，还在地上呢
        # for i in range(2, n + 1):
        #     dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])
        # return dp[-1]

    def uniquePaths(self, m: int, n: int) -> int:
        """ 62.不同路径
            中等 """
        # 时间：O(m*n) 空间：O(m*n)
        # dp = [[1] * n] + [[1] + [0] * (n - 1) for _ in range(m - 1)]    # 首行、首列初始化为1
        # for i in range(1, m):
        #     for j in range(1, n):
        #         dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        # return dp[-1][-1]

        # 优化空间复杂度 O(n)
        dp = [1] * n
        for i in range(1, m):
            for j in range(1, n):
                dp[j] += dp[j - 1]
        return dp[-1]

    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        """ 63.不同路径II
            中等 """
        # 时间：O(m*n) 空间：O(m*n)
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        dp = [[0] * n for _ in range(m)]
        for i in range(m):
            if obstacleGrid[i][0] == 1:
                break
            dp[i][0] = 1
        for j in range(n):
            if obstacleGrid[0][j] == 1:
                break
            dp[0][j] = 1
        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] == 1:     # 遇到障碍 跳过
                    continue
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[-1][-1]

        # 优化空间复杂度，不信整不了
        # m, n = len(obstacleGrid), len(obstacleGrid[0])
        # dp = [0] * n        # 一维dp
        # for j in range(n):
        #     if obstacleGrid[0][j] == 1:
        #         break
        #     dp[j] = 1
        # for i in range(1, m):
        #     for j in range(n):
        #         if obstacleGrid[i][j] == 1:
        #             dp[j] = 0
        #         elif j != 0:
        #             dp[j] += dp[j - 1]
        # return dp[-1]

    def integerBreak(self, n: int) -> int:
        """ 343.整数拆分
            中等 """
        # 时间复杂度：O(n^2) 空间复杂度：O(n)
        dp = [0] * (n + 1)        # dp[i] 拆分i所得最大乘积
        dp[2] = 1       # dp[0] dp[1]不符合dp定义，不使用
        for i in range(3, n + 1):
            for j in range(1, i // 2 + 1):      # 从i拆分一个j来
                dp[i] = max(dp[i], j * (i - j), j * dp[i - j])
        return dp[-1]

    def numTrees(self, n: int) -> int:
        """ 96.不同的二叉搜索树
            中等 """
        dp = [0] * (n + 1)
        dp[0] = 1
        dp[1] = 1
        for i in range(2, n + 1):
            for j in range(i):
                dp[i] += dp[j] * dp[i - 1 - j]
        return dp[-1]

    def test_2_wei_bag_problem1(self):
        """ 01背包 导读1 """
        weights = [1, 3, 4]
        values = [15, 20, 30]
        bagweight = 4

        dp = [[0] * (bagweight + 1) for _ in range(len(weights))]
        for j in range(1, bagweight + 1):
            if j >= weights[0]:
                dp[0][j] = values[0]
        # 遍历
        for i in range(1, len(weights)):
            for j in range(1, bagweight + 1):
                if j >= weights[i]:
                    dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weights[i]] + values[i])
                else:
                    dp[i][j] = dp[i - 1][j]
        print(dp[-1][-1])

    def test_1_wei_bag_problem(self):
        """ 01背包 导读2 """
        weights = [1, 3, 4]
        values = [15, 20, 30]
        bagweight = 4

        dp = [0] * (bagweight + 1)
        for i in range(len(weights)):
            for j in range(bagweight, weights[i] - 1, -1):
                dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
        print(dp[-1])

    def canPartition(self, nums: List[int]) -> bool:
        """ 416.分割等和子集
            中等
            01背包：先遍历物品、再遍历容量且倒序 """
        # 时间：O(n^2) 空间：O(n)
        if sum(nums) % 2:       # 注意 % / 的区别，竟然错在这
            return False
        bagweight = sum(nums) // 2
        n = len(nums)
        dp = [0] * (bagweight + 1)
        for i in range(n):
            for j in range(bagweight, nums[i] - 1, -1):
                dp[j] = max(dp[j], dp[j - nums[i]] + nums[i])
        return dp[-1] == bagweight

    def lastStoneWeightII(self, stones: List[int]) -> int:
        """ 1049.最后一块石头的重量II
            中等
            01背包 先遍历物品、再遍历容量，内层for倒序
            dp[sum/2]能装下多少石头; 就是想求容量为sum(stones)//2的背包能放下重量为多少的石头 """
        # 时间：O(mxn) 空间：O(m)
        target = sum(stones) // 2
        dp = [0] * (target + 1)
        for i in range(len(stones)):
            for j in range(target, stones[i] - 1, -1):
                dp[j] = max(dp[j], dp[j - stones[i]] + stones[i])
        return (sum(stones) - dp[-1]) - dp[-1]



if __name__ == "__main__":
    sl = Solution()

    stones = [31,26,33,21,40]
    print(sl.lastStoneWeightII(stones))
