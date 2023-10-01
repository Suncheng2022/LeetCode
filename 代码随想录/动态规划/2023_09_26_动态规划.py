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

    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        """ 494.目标和
            中等
            要能推导出 Left = (target+sum)/2，就能想到背包
            组合问题--dp[j]+=dp[j-nums[i]] """
        # 时间：O(mxn) 空间：O(n)
        if (target + sum(nums)) % 2:
            return 0
        left = (target + sum(nums)) // 2
        dp = [0] * (left + 1)       # dp[j] 装满容量j的背包有几种方法; 排列问题
        dp[0] = 1
        for i in range(len(nums)):
            for j in range(left, nums[i] - 1, -1):
                dp[j] += dp[j - nums[i]]
        return dp[-1]

    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        """ 474.一和零
            中等 """
        # 时间：O(mxn) 空间：O(mxn)
        from collections import Counter

        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for s in strs:
            zeroNum = Counter(s)['0']
            oneNum = Counter(s)['1']
            for i in range(m, -1, -1):
                for j in range(n, -1, -1):
                    if zeroNum <= i and oneNum <= j:
                        dp[i][j] = max(dp[i][j], dp[i - zeroNum][j - oneNum] + 1)
        return dp[-1][-1]

    def test_CompletePack(self):
        """ 完全背包 导读 """
        weights = [1, 3, 4]
        values = [15, 20, 30]
        bagweight = 4

        dp = [0] * (bagweight + 1)
        for i in range(len(weights)):
            for j in range(weights[i], bagweight + 1):
                dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
        print(dp[-1])

    def change(self, amount: int, coins: List[int]) -> int:
        """ 518.零钱兑换II
            中等
            硬币数无限--完全背包
                        1.遍历顺序均可，2.内层遍历正序
            求组合数--dp[j] += dp[j-weight[i]] """
        # 时间：O(nxm) 空间：O(n)
        n = len(coins)
        dp = [0] * (amount + 1)
        dp[0] = 1
        for i in range(n):
            for j in range(coins[i], amount + 1):       # 一定注意范围，好几次错在这里
                dp[j] += dp[j - coins[i]]
        return dp[-1]

    def combinationSum4(self, nums: List[int], target: int) -> int:
        """ 377.组合总和IV
            中等
            元素个数无限--完全背包
            求组合个数 且 顺序不同视为不同结果--即排列，外层for容量、内层for物品且正序
            求排列、组合递推公式--dp[j] += dp[j - nums[i]] """
        # 时间：O(target * n) 空间：O(target)
        n = len(nums)
        dp = [0] * (target + 1)
        dp[0] = 1
        for j in range(1, target + 1):
            for i in range(n):
                if j >= nums[i]:
                    dp[j] += dp[j - nums[i]]
        return dp[-1]

    def climbStairsPro(self, n: int) -> int:
        """ 70.爬楼梯(进阶)
            简单
            之前做过 现在使用背包做
            元素不限--完全背包
            顺序不同的结果视为不同--求排列 外层for容量、内层for物品 """
        # 时间：O(nxm) 空间：O(n)
        if n == 1:
            return 1
        dp = [0] * (n + 1)
        dp[1] = 1
        dp[2] = 2
        weights = [1, 2]
        for i in range(3, n + 1):
            for j in range(len(weights)):
                dp[i] += dp[i - weights[j]]
        return dp[-1]

    def coinChange(self, coins: List[int], amount: int) -> int:
        """ 322.零钱兑换
            中等
            元素无限--完全背包
            不求 排列 or 组合，所以 外层for物品、内层for容量且正序
            求最小硬币个数，递推公式应为dp[j] = min(dp[j], dp[j-weight[i] + 1]) """
        # 时间：O(nxm) 空间：O(m)
        n = len(coins)
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0       # 注意 dp 意义
        for i in range(n):
            for j in range(1, amount + 1):
                if j >= coins[i]:
                    dp[j] = min(dp[j], dp[j - coins[i]] + 1)
        return dp[-1] if dp[-1] != float('inf') else -1


if __name__ == "__main__":
    sl = Solution()

    coins = [2]
    amount = 3
    print(sl.coinChange(coins, amount))
