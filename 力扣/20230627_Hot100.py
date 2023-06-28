from typing import List, Optional


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        """
        739.每日温度
        2023.06.27 中等
        题解：递减栈
        """
        stack = []      # 维护递减栈
        n = len(temperatures)
        res = [0] * n
        for i in range(n):
            if not stack or temperatures[i] <= stack[-1][-1]:
                stack.append([i, temperatures[i]])
            # 当temperatures[i] > stack[-1][-1]时，肯定是栈中元素第一次遇到较大元素
            while stack and temperatures[i] > stack[-1][-1]:
                ind, _ = stack.pop()
                res[ind] = i - ind      # 栈顶元素第一次遇见较大元素，计算间隔天数
            stack.append([i, temperatures[i]])
        return res

    def countSubstrings(self, s: str) -> int:
        """
        647.回文子串
        2023.06.27 中等
        题解：向两端探测，稍作修改，因为题目要求位置不同即算视作不同子串
        """
        res = set()     # set可去重
        for i in range(len(s)):
            res.add((i))        # 本身也是回文子串
            l, r = i - 1, i + 1
            while l >= 0 and s[l] == s[i]:      # 向左探测所有与s[i]相同的元素
                res.add((l, i))
                l -= 1
            while r < len(s) and s[r] == s[i]:  # 向右探测所有与s[i]相同的元素
                res.add((i, r))
                r += 1
            while l >= 0 and r < len(s):
                if s[l] == s[r]:
                    res.add((l, r - 1))
                    l -= 1
                    r += 1
                else:
                    break
        return res.__len__()

    def leastInterval(self, tasks: List[str], n: int) -> int:
        """
        621.任务调度器
        2023.06.27 中等
        题解：res = (max_freq - 1) * (n + 1) + num of max_freq
        """
        from collections import Counter

        task2counts = sorted(Counter(tasks).items(), key=lambda x: x[1], reverse=True)
        max_freq = task2counts[0][-1]
        num_max_freq = sum([True for _, counts in task2counts if counts == max_freq])
        # 最后加num_max_freq，因为还有自己，所以一共num_max_freq个
        res = (max_freq - 1) * (n + 1) + num_max_freq
        return res if res >= len(tasks) else len(tasks)

    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
        """
        617.合并二叉树
        2023.06.27 简单
        题解：回忆答案 递归
        """
        # 递归 前序遍历
        # def dfs(root1, root2):
        #     if not (root1 and root2):       # 至少1个空
        #         return root1 if root1 else root2
        #     root1.val += root2.val
        #     root1.left = dfs(root1.left, root2.left)
        #     root1.right = dfs(root1.right, root2.right)
        #     return root1
        #
        # return dfs(root1, root2)

        # 迭代 广度优先遍历
        # 广度优先需要辅助存储了
        if not (root1 and root2):   # 只要有1个空
            return root1 if root1 else root2
        queue = [[root1, root2]]
        while queue:
            node1, node2 = queue.pop(0)
            node1.val += node2.val
            if node1.left and node2.left:   # node1.left、node2.left都不空直接入队
                queue.append([node1.left, node2.left])
            elif not node1.left:            # node1.left空、node2.left不空，拷贝过去；node1.left不空、node2.left空，不用处理，反正最后返回root1
                node1.left = node2.left
            if node1.right and node2.right:     # node1.right、node2.right同理
                queue.append([node1.right, node2.right])
            elif not node1.right:
                node1.right = node2.right
        return root1        # 最终返回root1

    def findUnsortedSubarray(self, nums: List[int]) -> int:
        """
        581.最短无序连续子数组
        2023.06.27 中等
        题解：从左往右找 非升序 最大值，从右往左找 非升序 最小值
        """
        max_num, right = float('-inf'), -1
        min_num, left = float('inf'), -1
        n = len(nums)
        for i in range(n):
            if nums[i] < max_num:
                right = i       # 从左往右，遇到那段 非升序 就更新right；最终指向 非升序 的最右边元素
            else:
                max_num = nums[i]
            if nums[n - 1 - i] > min_num:
                left = n - 1 - i    # 从右往左，遇到那段 非升序 就更新right；最终指向 非升序 的最左边元素
            else:
                min_num = nums[n - 1 - i]
        return right - left + 1 if right != left else 0     # 可能不需要调整；注意返回的是 长度

    def subarraySum(self, nums: List[int], k: int) -> int:
        """
        560.和为K的子数组
        2023.06.27 中等
        题解：前缀和。当前的前缀和presum，前缀和presum-k出现的次数
        """
        from collections import defaultdict

        # 前缀和及其出现次数
        preSums = defaultdict(int)
        preSums[0] = 1
        res = 0
        presum = 0      # 当前的前缀和
        for i in range(len(nums)):
            presum += nums[i]       # 遍历，当前的前缀和
            # 是否有前缀和为presum-k的，并累加出现次数；
            # 【当前的前缀和presum是从头一个元素一个元素遍历累加得来的，肯定会在某个时刻超过k；我们不断寻找前面是否出现过大小为presum-k的前缀和，二者的距离就是k】
            res += preSums[presum - k]
            preSums[presum] += 1    # 当前的前缀和出现次数+1
        return res

    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        """
        543.二叉树的直径
        2023.06.27 中等
        题解：递归，求树的高度；递归过程不断更新 路径
        """
        def depth(node):
            """ 计算根节点为node时数的高度 """
            if not node:
                return 0
            l = depth(node.left)
            r = depth(node.right)
            nonlocal res
            res = max(res, l + r)       # 更新 路径，不是计算树的高度，所以不用 +1
            return max(l, r) + 1       # 树的高度要加根节点自己，即+1

        res = 0
        depth(root)
        return res

    def convertBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """
        538.把二叉搜索树转换为累加树
        2023.06.28 中等
        题解：能想到 右根左 的递归遍历顺序
        """

        # *序遍历似乎都是：开始判断一下--递归终止条件；在遍历根节点时操作
        def dfs(node):
            if not node:
                return 0
            dfs(node.right)
            nonlocal total
            total += node.val
            node.val = total
            dfs(node.left)

        total = 0
        dfs(root)
        return root

    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        """
        494.目标和
        2023.06.28 中等
        题解：数组元素均非负，数组和sum，带负号元素和绝对值neg，正号元素和绝对值sum-neg，因此有sum-neg-neg=target-->(sum-target)//2=neg，求能组成neg的方案数
        """
        sum_nums = sum(nums)
        if sum_nums < target or (sum_nums - target) % 2:    # 如果数组和小于target、数组和-target不是偶数，那么没有方案
            return 0
        neg = (sum_nums - target) // 2
        n = len(nums)
        dp = [[0] * (neg + 1) for _ in range(n + 1)]    # dp[i][j]: nums前i个数能选出和为j的方案数
        dp[0][0] = 1    # dp[0][0]=1, dp[0][>=1]均为0
        for i in range(1, n + 1):       # 【0行已初始化，这里不再计算】！！！
            for j in range(neg + 1):
                if nums[i - 1] > j:    # 不能选nums[i]
                    dp[i][j] = dp[i - 1][j]     # i-1表达了不选nums[i]
                else:
                    dp[i][j] = dp[i - 1][j] + dp[i - 1][j - nums[i - 1]]    # j-nums[i]表达了必选nums[i]
        return dp[-1][-1]

    def hammingDistance(self, x: int, y: int) -> int:
        """
        461.汉明距离
        2023.06.28 简单
        """
        tmp = x ^ y
        res = 0
        while tmp:
            if tmp & 1:
                res += 1
            tmp >>= 1
        return res


if __name__ == '__main__':
    sl = Solution()
    nums = [1, 1, 1, 1, 1]
    target = 3
    print(sl.findTargetSumWays(nums, target))