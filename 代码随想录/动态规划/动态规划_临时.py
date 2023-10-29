class Solution:
    def test_2_wei_bag_problem1(self):
        """ 01背包导读1 二维数组实现 """
        weights = [1, 3, 4]
        values = [15, 20, 30]
        bagweight = 4

        # 定义dp
        dp = [[0] * (bagweight + 1) for _ in range(len(values))]
        # 初始化dp
        for i in range(len(values)):
            dp[i][0] = 0
        for j in range(bagweight + 1):
            dp[0][j] = values[0] if j >= weights[0] else 0
        # 遍历：二维dp遍历顺序均可；
        #      先遍历物品、再遍历容量比较好理解
        for i in range(1, len(weights)):        # 使用 索引0物品 已经用于初始化了
            for j in range(bagweight + 1):
                if j < weights[i]:              # 不放 weights[i]
                    dp[i][j] = dp[i - 1][j]
                else:                           # 放 weights[i]
                    dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weights[i]] + values[i])
        return dp[-1][-1]

    def test_1_wei_bag_problem(self):
        """ 01背包导读2 一维数组实现 """
        weights = [1, 3, 4]
        values = [15, 20, 30]
        bagweight = 4

        # 定义一维dp
        dp = [0] * (bagweight + 1)
        # 遍历：一维dp，必须先遍历物品、再遍历容量且倒序
        for i in range(len(weights)):
            for j in range(bagweight, weights[i] - 1, -1):       # dp[0] = 0, 这是包含在初始化中的
                dp[j] = max(dp[j], dp[j - weights[i]] + values[i])     # 一维dp去掉的是维度i
        return dp[-1]

if __name__ == '__main__':
    sl = Solution()
    print(sl.test_1_wei_bag_problem())

"""
动归五部曲：
    1.确定dp数组及下标的意义
    2.递推公式
    3.初始化dp
    4.确定遍历顺序
    5.手动模拟dp
"""