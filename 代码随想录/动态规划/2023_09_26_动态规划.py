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

if __name__ == "__main__":
    sl = Solution()

    n = 3
    print(sl.climbStairs(n))
