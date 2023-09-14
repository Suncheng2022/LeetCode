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

    def test_2_wei_bag_problem1(self):
        """ 01背包问题 导读
            二维dp dp[i][j] 从0~i的物品中选 放入容量为j的背包，所得最大价值 """
        weights = [1, 3, 4]
        values = [15, 20, 30]
        bagweight = 4
        # 答案应为35

        n = len(weights)
        dp = [[0] * (bagweight + 1) for _ in range(n)]
        for j in range(bagweight + 1):      # 初始化首行
            if j >= weights[0]:
                dp[0][j] = values[0]
        # 首列需要初始化为0 dp[i][0]=0，因为包的容量为0，啥也放不下。可省略
        # 二维dp先遍历谁都行，这里先遍历物品、再遍历背包容量
        for i in range(1, n):
            for j in range(weights[i], bagweight + 1):
                dp[i][j] = max(dp[i - 1][j],                            # 不放weights[i]
                               dp[i - 1][j - weights[i]] + values[i])   # 放weigits[i]
        print(dp[-1][-1])

    def test_1_wei_bag_problem(self):
        """ 01背包问题 导读
            一维dp dp[j] 去掉了i维度，表示：容量为j的背包能装下的最大价值
             仔细看《代码随想录》讲解，一维dp[j]是将上一层即i-1层复制到i层，递推公式为dp[j]=max(dp[j], dp[j-weights[i]]+values[i]),
             其实本质上还是取 左、上的最大值 """
        weights = [1, 3, 4]
        values = [15, 20, 30]
        bagweight = 4
        # 答案应为35

        dp = [0] * (bagweight + 1)      # dp[j] 背包容量j能装下的最大价值
        dp[0] = 0       # 初始化，强调一下吧
        # 一维dp，先遍历物品、再遍历背包容量且倒序；
        # 先遍历背包容量则遍历物品时每次只会放进1个物品，这是不对的
        # 正序遍历容量会导致物品放入2次，因为推导公式dp[j]=max(dp[j], dp[j-weights[i]]+values[i])可以看出每次都要用到前面的dp，正序会导致物品重复放入
        for i in range(len(weights)):       # 这里起始索引为0，0号物品也要取！
            for j in range(bagweight, weights[i] - 1, -1):
                dp[j] = max(dp[j],                              # 其实是取二维的dp[i-1][j]
                            dp[j - weights[i]] + values[i])     # 其实是取二维的dp[i-1][j-weights[i]] + values[i]
        print(dp[-1])

    def canPartition(self, nums: List[int]) -> bool:
        """ 416.分割等和子集
            竟然也能想到用背包问题，dp[i] 容量为i的背包能盛下的最大价值 """
        if sum(nums) % 2:
            return False
        target = sum(nums) // 2
        dp = [0] * (target + 1)
        # 一维dp 先遍历物品、再遍历容量且倒序；
        # 遍历顺序：若先遍历容量、再遍历物品，会导致每次只有1个物品放入
        # 遍历容量倒序：若正序，根据递推公式，会导致每个物品放入2次，01背包不能重复放入
        for i in range(len(nums)):
            for j in range(target, nums[i] - 1, -1):
                dp[j] = max(dp[j],                          # 不放入nums[i]；即二维dp的dp[i-1][j]
                            dp[j - nums[i]] + nums[i])      # 放入nums[i]
        return dp[-1] == target

    def lastStoneWeightII(self, stones: List[int]) -> int:
        """ 1049.最后一块石头的重量II
            类似 416.分割等和子集 只不过本题不是问的能不能恰好装下一半 """
        target = sum(stones) // 2
        dp = [0] * (target + 1)     # dp[i] 容量为i的背包所能装下的最大价值
        dp[0] = 0
        # 01背包一维dp 先遍历物品、再遍历容量且倒序
        # 先遍历物品：若先遍历容量，会导致每次只装入1个物品
        # 遍历容量倒序：若正序，由递推公式 物品会装入2次
        for i in range(len(stones)):
            for j in range(target, stones[i] - 1, -1):
                dp[j] = max(dp[j], dp[j - stones[i]] + stones[i])
        return sum(stones) - dp[-1] - dp[-1]        # 返回2堆之差

    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        """ 494.目标和
            重要的是能想到 left=(sum(nums) + target) / 2
            之前01背包是求 装满容量j的背包所得最大价值，本题是求装满容量j的背包有几种方式，其实是组合问题——dp[j] += dp[j-nums[i]]
            【记住】求装满背包有几种方法时，递推公式一般为dp[j]+=dp[j-nums[i]] """
        if (sum(nums) + target) % 2 or abs(target) > sum(nums):
            return 0
        target = (sum(nums) + target) // 2
        dp = [0] * (target + 1)    # dp[j] 装满容量j的背包有几种方法
        dp[0] = 1       # dp[0]是递推公式的起点；若nums=[0], target=0, 只有1种方法，无论元素前添加的是加号还是减号
        for i in range(len(nums)):
            for j in range(target, nums[i] - 1, -1):
                dp[j] += dp[j - nums[i]]
        return dp[-1]

    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        """ 474.一和零
            将0的个数、1的个数视为物品的重量 """
        from collections import Counter

        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for s in strs:      # 遍历物品
            zeroNum = Counter(s)['0']
            oneNum = Counter(s)['1']
            # 两个for 遍历容量
            for i in range(m, zeroNum - 1, -1):
                for j in range(n, oneNum - 1, -1):
                    dp[i][j] = max(dp[i][j],                            # 相当于二维dp的 dp[i-1][j]
                                   dp[i - zeroNum][j - oneNum] + 1)     # 相当于二维dp的 dp[i-1][j-weights[i]] + values[i]
        return dp[-1][-1]

        # 再写一遍
        # from collections import Counter
        #
        # dp = [[0] * (n + 1) for _ in range(m + 1)]      # dp[i][j] i个0、j个1能组成的最大子集
        # for s in strs:
        #     zeroNum = Counter(s)['0']
        #     oneNum = Counter(s)['1']
        #     for i in range(m, zeroNum - 1, -1):
        #         for j in range(n, oneNum - 1, -1):
        #             dp[i][j] = max(dp[i][j], dp[i - zeroNum][j - oneNum] + 1)
        # return dp[-1][-1]

    def test_CompletePack(self):
        """ 完全背包 导读
            完全背包 与 01背包 的唯一区别，遍历容量顺序不同[01背包要倒序，完全背包要正序]
            完全背包内外层循环先后顺序均可，01背包内外层循环只能外层物品内层容量 """
        weights = [1, 3, 4]
        values = [15, 20, 30]
        bagweight = 4
        # 答案应为60

        dp = [0] * (bagweight + 1)
        for i in range(len(weights)):
            for j in range(weights[i], bagweight + 1):
                dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
            print(dp)
        print(dp[-1])


if __name__ == '__main__':
    sl = Solution()

    strs = ["10", "0001", "111001", "1", "0"]
    m = 5
    n = 3
    print(sl.test_CompletePack())
