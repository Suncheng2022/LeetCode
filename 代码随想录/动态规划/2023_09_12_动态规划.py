from typing import List, Optional


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

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

    def change(self, amount: int, coins: List[int]) -> int:
        """ 518.零钱兑换II
            硬币可重复用--完全背包
            求组合数--dp[j]+=dp[j-nums[i]]
        完全背包：1.内外遍历顺序都可以【但本题不可以，只能外层物品、内层容量——1.1求组合数 外层物品、内层容量 1.2求排列数 外层容量、内层物品】
                2.遍历容量必须正序，以保证物品可重复放入
        01背包：1.外层遍历物品、内层遍历容量 2.内层遍历容量必须倒序，以保证物品不重复放入 """
        dp = [0] * (amount + 1)     # dp[i] 组成金额i，有几种方式
        dp[0] = 1
        for i in range(len(coins)):
            for j in range(coins[i], amount + 1):
                dp[j] += dp[j - coins[i]]
        return dp[-1]

    def combinationSum4(self, nums: List[int], target: int) -> int:
        """ 377.组合总和IV
            元素可重复使用--完全背包 内层for正序(似乎无论内层遍历是容量还是物品)
            顺序不同视为不同组合--求排列 外层容量、内层物品，若外层物品、内层容量，若nums={1,3,4} 则3只会出现在1后面 """
        dp = [0] * (target + 1)     # dp[i] 组成和为i有几种方法
        dp[0] = 1
        for i in range(1, target + 1):
            for j in range(len(nums)):
                if i >= nums[j]:
                    dp[i] += dp[i - nums[j]]
        return dp[-1]

    def climbStairs(self, n: int) -> int:
        """ 70.爬楼梯（进阶版）
            可转化为背包问题，物品--每次爬1个台阶、每次爬2个台阶 容量--需要爬n阶
            物品可重复使用--完全背包，内层遍历正序(似乎无论内层遍历的什么)
            根据题意，是有顺序的--外层容量、内层物品
            题目求有多少种方法，即装满容量为n的背包有多少种方法--递推公式dp[j]+=dp[j-nums[i]] """
        dp = [0] * (n + 1)      # dp[i] 装满容量i的背包有多少种方法
        dp[0] = 1       # 初始化
        nums = [1, 2]   # 相当于物品
        for j in range(n + 1):              # 遍历容量
            for i in range(len(nums)):      # 遍历物品
                if j >= nums[i]:
                    dp[j] += dp[j - nums[i]]
        return dp[-1]

    def coinChange(self, coins: List[int], amount: int) -> int:
        """ 322.零钱兑换
            硬币数量无限--完全背包，内层for正序(无论内层遍历什么)
            装满背包的最少方法--那就将每个物品价值视为1(即个数)；
                            递推公式能想到是dp[j]=dp[j-nums[i]]+1, 但没表达'最少个数'，所以完整递推公式应为dp[j]=min(dp[j],dp[j-nums[i]]+1) """
        dp = [float('inf')] * (amount + 1)     # 根据递推特性，求最小值，所以初始化为最大的数；dp[i] 组成金额i所需最少硬币数 / 装满容量为i的背包所需最小物品数量，我觉得我解释是对的
        dp[0] = 0
        for i in range(len(coins)):
            for j in range(coins[i], amount + 1):
                dp[j] = min(dp[j], dp[j - coins[i]] + 1)        # 装满容量j的背包所需最少物品数
        return dp[-1] if dp[-1] != float('inf') else -1

    def numSquares(self, n: int) -> int:
        """ 279.完全平方数
            元素可重复使用--完全背包 内层for正序（无论内层遍历什么）
            装满背包所需最少物品数--递推公式 dp[j]=min(dp[j],dp[j-nums[i]] + 1) """
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        for i in range(1, int(pow(n, .5)) + 1):     # 这里若写成n//2 + 1，取不到i=1，计算n=1时会错误；物品最大也是n开方嘛
            for j in range(i * i, n + 1):
                dp[j] = min(dp[j], dp[j - i * i] + 1)
        return dp[-1] if dp[-1] != float('inf') else -1

    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        """ 139.单词拆分
            物品--字典的单词，背包--字符串s；s能否被装满，即背包问题
            物品可重复使用--完全背包，内层for正序(无论内层遍历什么)
            题目隐含了顺序问题--先遍历容量、再遍历物品 """
        n = len(s)
        dp = [False] * (n + 1)      # dp[i] 前i个字符能否被字典的单词表示
        dp[0] = True    # 纯为递推公式
        for i in range(1, n + 1):
            for j in range(i):      # 遍历物品没写正确，本题并不是遍历wordDict
                if dp[j] and s[j:i] in wordDict:       # j指示当前遍历s的下标，dp[j:i]是否能被字典表示
                    dp[i] = True
        return dp[-1]

    def rob(self, nums: List[int]) -> int:
        """ 198.打家劫舍 """
        # 《代码随想录》感觉dp更清晰
        n = len(nums)
        if n <= 2:
            return max(nums)
        dp = [0] * n    # dp[i] 截止到索引i房屋能偷到的最大金额
        dp[0] = nums[0]
        dp[1] = max(nums[:2])
        for i in range(2, n):
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
        return dp[-1]

        # n = len(nums)
        # dp = [0] * (n + 1)      # dp[i] 前i个房屋能偷到的最大金额
        # dp[1] = nums[0]
        # for i in range(2, n + 1):
        #     dp[i] = max(dp[i - 1], dp[i - 2] + nums[i - 1])
        # return dp[-1]

    def robII(self, nums: List[int]) -> int:
        """ 213.打家劫舍II """
        # 《代码随想录》边界条件就比较简洁了
        def rob_(nums, start, end):
            """ start end均包含 """
            if start == end:
                return nums[start]
            dp = [0] * len(nums)
            # 看人家这初始化，避免了自己那样复杂的边界处理
            dp[start] = nums[start]
            dp[start + 1] = max(nums[start:start + 2])
            for i in range(start + 2, end + 1):
                dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
            return dp[end]

        # 特殊处理长度为1，则rob_刚好可以处理len(nums)>=2的情况
        if len(nums) == 1:      # 题目说明nums长度至少为1
            return nums[0]
        res1 = rob_(nums, 0, len(nums) - 2)
        res2 = rob_(nums, 1, len(nums) - 1)
        return max(res1, res2)

        # 自己实现的，逻辑太复杂，尤其边界条件
        # def rob_(nums, start, end):
        #     """ 198.打家劫舍 的逻辑
        #         start、end均为包含元素 """
        #     nums = nums[start:end + 1][:]
        #     n = len(nums)
        #     dp = [0] * n    # dp[i] 截止到索引i房间(包括)，能偷到的最大金额
        #     dp[0] = nums[0]
        #     dp[1] = max(nums[:2])
        #     for i in range(2, n):
        #         dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
        #     return dp[-1]
        #
        # # 成环的 打家劫舍 逻辑
        # n = len(nums)
        # if n <= 2:
        #     return max(nums)
        # res1 = rob_(nums, 0, len(nums) - 2)     # 不考虑尾元素
        # res2 = rob_(nums, 1, len(nums) - 1)     # 不考虑首元素
        # return max(res1, res2)

    def rob(self, root: Optional[TreeNode]) -> int:
        """ 337.打家劫舍III
            回溯递归 + 动态规划"""
        def rob_(node):
            # 递归终止条件
            if not node:
                return [0, 0]
            lefts = rob_(node.left)
            rights = rob_(node.right)
            val0 = max(lefts) + max(rights)             # 不偷当前节点
            val1 = node.val + lefts[0] + rights[0]      # 偷当前节点
            return [val0, val1]

        return max(rob_(root))

    def maxProfit(self, prices: List[int]) -> int:
        """ 121.买卖股票的最佳时机
            只一次买卖，因此'买入'的时候只能是-prices[i]，而不是 昨天持有-prices[i] """
        # 《代码随想录》动态规划思路，状态转移，把系列题目串起来
        n = len(prices)
        dp = [[0, 0] for _ in range(n)]    # dp[i] 索引第i天的 不持有/持有 的最大收益
        dp[0] = [0, -prices[0]]         # 初始化索引第0天
        for i in range(1, n):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
            dp[i][1] = max(dp[i - 1][1], -prices[i])    # 注意 昨天不持有 转移到 今天持有 的状态转移
        return dp[-1][0]

        # 自己写的，想了一会；还是没有系统的解题思路，还是得《代码随想录》
        # res = 0
        # minPrice = float('inf')
        # for i in range(1, len(prices)):
        #     minPrice = min(minPrice, prices[i - 1])
        #     res = max(res, prices[i] - minPrice)
        # return res

    def maxProfitII(self, prices: List[int]) -> int:
        """ 122.买卖股票的最佳时机II
            可多次买卖，因此 相比 121.买卖股票的最佳时机，'买入'的时候是 昨天持有-prices[i]了，因为可多次买卖，买入之前可能有些盈利累计了 """
        n = len(prices)
        dp = [[0, 0] for _ in range(n)]     # dp[i] 索引第i天的 不持有/持有 的最大收益
        dp[0] = [0, -prices[0]]         # 初始化索引第0天
        for i in range(1, n):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])
        return dp[-1][0]

    def maxProfitIII(self, prices: List[int]) -> int:
        """ 123.买卖股票的最佳时机III
            至多买卖2次，状态多了而已
            状态：
                状态0：无操作(可不写)
                状态1：第一次持有
                状态2：第一次不持有
                状态3：第二次持有
                状态4：第二次不持有 """
        n = len(prices)
        dp = [[0] * 5 for _ in range(n)]
        dp[0] = [0, -prices[0], 0, -prices[0], 0]
        for i in range(1, n):
            # 我就没有考虑状态0的计算，也能AC
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])
            dp[i][2] = max(dp[i - 1][2], dp[i - 1][1] + prices[i])
            dp[i][3] = max(dp[i - 1][3], dp[i - 1][2] - prices[i])
            dp[i][4] = max(dp[i - 1][4], dp[i - 1][3] + prices[i])
        return max(dp[-1])

    def maxProfitIV(self, k: int, prices: List[int]) -> int:
        """ 188.买卖股票的最佳时机IV
            至多买卖k次，状态再多些而已 """
        n = len(prices)
        dp = [[0] * (2 * k + 1) for _ in range(n)]      # 索引第i天的 第几次股票持有或不持有 所得最大收益
        # 初始化索引第0天dp
        for i in range(1, 2 * k + 1):
            if i % 2:   # 奇数 代表 第某次持有
                dp[0][i] = -prices[0]
            else:       # 偶数 代表 第某次不持有
                dp[0][i] = 0
        for i in range(1, n):
            for j in range(1, 2 * k + 1):
                if j % 2:   # 奇数 持有
                    dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - 1] - prices[i])
                else:       # 偶数 不持有
                    dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - 1] + prices[i])
        return dp[-1][-1]   # 同 max(dp[-1])

    def maxProfitFreeze(self, prices: List[int]) -> int:
        """ 309.买卖股票的最佳时机含冷冻期
            含冷冻期，依然是设置状态
                状态0：持有
                状态1：不持有(在冷冻期)
                状态2：不持有(不在冷冻期) """
        n = len(prices)
        dp = [[0, 0, 0] for _ in range(n)]
        dp[0] = [-prices[0], 0, 0]
        for i in range(1, n):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][2] - prices[i])
            dp[i][1] = dp[i - 1][0] + prices[i]
            dp[i][2] = max(dp[i - 1][2], dp[i - 1][1])
        return max(dp[-1])

    def maxProfitFee(self, prices: List[int], fee: int) -> int:
        """ 714.买买股票的最佳时机含手续费
            相比 122.买卖股票的最佳时机II 仅多了手续费而已 """
        n = len(prices)
        dp = [[0, 0] for _ in range(n)]
        dp[0] = [0, -prices[0]]
        for i in range(1, n):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i] - fee)
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])
        return dp[-1][0]    # 同 max(dp[-1])

    def lengthOfLIS(self, nums: List[int]) -> int:
        """ 300.最长递增子序列
            不是你想的那么简单滴
            如果nums[j] < nums[i], 则nums[i]可以放到nums[j]后面-->nums[i]能放到哪些nums[j]后面，需要遍历j了，从0到i-1 """
        n = len(nums)
        dp = [1] * n
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i]:    # 如果nums[i]可以放到这个nums[j]后面，则dp[i]更新为dp[j]+1
                    dp[i] = max(dp[i], dp[j] + 1)
            print(dp)
        return max(dp)      # 这里不能返回dp[-1]，因为以nums[-1]结尾的不一定是最长的子序列，万一中间有元素比最后的元素大呢，对不对。这题并不简单呀

    def findLengthOfLCIS(self, nums: List[int]) -> int:
        """ 674.最长连续递增序列 """
        n = len(nums)
        dp = [1] * n    # 以nums[i]结尾的最长连续递增序列的长度为 dp[i]
        for i in range(1, n):
            if nums[i - 1] < nums[i]:
                dp[i] = dp[i - 1] + 1
        return max(dp)      # 注意，不是返回dp[-1]，自己想想

    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        """ 718.最长重复子数组
            竟然是求 最长连续公共子序列，注意是 连续 """
        m, n = len(nums1), len(nums2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]      # dp[i][j] 以nums1[i-1]结尾的连续子序列、以nums2[j]结尾的连续子序列 的 连续公共子序列长度
        # dp 索引有0的似乎不参与计算，只参与推导，因为dp有0没意义 初始化为0即可
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if nums1[i - 1] == nums2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1     # 注意，是求 连续子序列 的长度
        return max(max(ls) for ls in dp)

    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        """ 1143.最长公共子序列
            dp定义与 718.最长重复子数组 是不同的 """
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]      # dp[i][j] 索引从 0~i-1 的text1、索引从 0~j-1 的text2 的 公共子序列长度；并非text1[i-1] text2[j-1]结尾哈
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # 这里计算dp[i][j]要分情况了：text1[i-1]==text2[j-1] 或 text1[i-1]!=text2[j-1]
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])      # 之所以有这句，看dp定义；看dp推导图，dp[i][j]的左上角都会<=左、上
        return dp[-1][-1]       # 现在结果是dp[-1][-1]了，因为从左上、上、左三个方向最大值推导来的

    def maxUncrossedLines(self, nums1: List[int], nums2: List[int]) -> int:
        """ 1035.不相交的线
            即 最长公共子序列，'不连续' """
        m, n = len(nums1), len(nums2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]      # dp[i][j] 索引0~i-1的nums1、索引0~j-1的nums2 的 公共子序列长度
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if nums1[i - 1] == nums2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[-1][-1]

    def maxSubArray(self, nums: List[int]) -> int:
        """ 53.最大子数组和
            题目包含求'连续'元素时，dp[i]似乎都要定义为 以nums[i]为结尾的 ... """
        n = len(nums)
        dp = [0] * n    # dp[i] 以nums[i]结尾的连续子数组的和
        dp[0] = nums[0]
        for i in range(1, n):   # 我多写了判断条件 if nums[i] > 0 ，导致dp不能全部更新
            dp[i] = max(dp[i - 1] + nums[i], nums[i])
        return max(dp)      # 不一定是dp[i]最大，你想，中间达到最大 后面都是负数，对吧

    def isSubsequence(self, s: str, t: str) -> bool:
        """ 392.判断子序列
            自己没想通递推公式，推一遍dp图清爽很多
            很像 1143.最长公共子序列：分2种情况
                1.nums1[i-1]==nums2[j-1]则dp[i][j]=dp[i-1][j-1]+1
                2.nums1[i-1]!=nums2[j-1]则只放一边（1143.是两边都放试试，挑大的）"""
        m, n = len(s), len(t)
        # dp[i][j] 以s[i-1]结尾、以t[j-1]结尾 的 公共子序列长度
        # 初始化 dp索引带0的均为0，因为0表示空子串
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s[i - 1] == t[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = dp[i][j - 1]     # 只删t 的与s不相等的元素，因为是判断s是否为t的子序列
        return dp[-1][-1] == m


if __name__ == '__main__':
    sl = Solution()

    s = "axc"
    t = "ahbgdc"
    print(sl.isSubsequence(s, t))
