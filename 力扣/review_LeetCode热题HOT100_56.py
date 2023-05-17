from typing import List, Optional


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        """
        105.从前序与中序遍历序列构造二叉树
        中等 2023.03.15
        题解：https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/solution/cong-qian-xu-yu-zhong-xu-bian-li-xu-lie-gou-zao-9/
        :param preorder:先序遍历序列
        :param inorder:中序遍历序列
        :return:
        """
        index = {val: i for i, val in enumerate(inorder)}  # 哈希映射，通过val使用o(1)时间复杂度定位到索引i

        def build_sub_tree(preorder_left, preorder_right, inorder_left, inorder_right):
            """ build_sub_tree参数指示的同一子树 的 不同遍历顺序 """
            # 递归停止条件
            if preorder_left > preorder_right:
                return None
            # 定位根节点在先序遍历、中序遍历的索引
            preorder_root = preorder_left
            inorder_root = index[preorder[preorder_left]]
            # 建立根节点
            root_node = TreeNode(preorder[preorder_left])
            # 计算左子树长度，其实也隐含了右子树长度，所以只计算一个子树的长度即可，计算左子树方便些
            size_left_subtree = inorder_root - inorder_left
            # 构建左右子树
            root_node.left = build_sub_tree(preorder_left + 1, preorder_left + size_left_subtree, inorder_left,
                                            inorder_root - 1)
            root_node.right = build_sub_tree(preorder_left + size_left_subtree + 1, preorder_right, inorder_root + 1,
                                             inorder_right)
            return root_node

        n = len(index)
        return build_sub_tree(0, n - 1, 0, n - 1)

    def maxDepth(self, root: Optional[TreeNode]) -> int:
        """
        104.二叉树的最大深度
        简单 2023.03.16
        题解：
        :param root:
        :return:
        """
        if not root:  # 考虑到后面一次处理一层，所以queue里面不能有空节点，所以在此判定一下再初始化queue
            return 0
        queue = [root]
        res = []
        while queue:
            cur_layer_num = len(queue)
            tmp = []
            for i in range(cur_layer_num):
                node = queue.pop(0)
                tmp.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(tmp)
        return len(res)

    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        """
        102.二叉树的层序遍历
        中等 2023.03.18
        题解：思路可用于104.二叉树的最大深度
        :param root:
        :return:
        """
        if not root:
            return []
        queue = [root]
        res = []
        while queue:
            sizes_per_layer = len(queue)
            nodes_per_layer = []
            for _ in range(sizes_per_layer):
                node = queue.pop(0)
                nodes_per_layer.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(nodes_per_layer)
        return res

    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        """
        101.对称二叉树
        简单 2023.03.18
        题解：基于队列
        :param root:
        :return:
        """
        if not (root.left or root.right):  # 题目说几点数目>=1，所以这里只判断root无根节点
            return True
        queue = [root.left, root.right]  # 考虑放进来的2个节点，可能：都空，1个空，都不空。在while中依次判断
        while queue:
            left = queue.pop(0)
            right = queue.pop(0)
            if not (left or right):  # 判断都空吗
                continue
            if not (left and right) or (left.val != right.val):  # 首先检查left、right是否只有1个空 或 二者值不等，这都是不对称的
                return False
            queue.extend([left.left, right.right, left.right, right.left])
        return True

    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        """
        98.验证二叉搜索树
        中等 2023.03.19
        题解：while循环，尽量不用递归 不好懂
        :param root:
        :return:
        """
        # 重写一遍
        stack = []
        inorder = float('-inf')
        while stack or root:
            while root:
                stack.append(root)
                root = root.left
            node = stack.pop()
            if node.val <= inorder:
                return False
            inorder = node.val
            root = node.right
        return True
        # stack = []      # 栈 先进后出
        # previous = float('-inf')
        # while stack or root:
        #     while root:
        #         stack.append(root)
        #         root = root.left
        #     node = stack.pop()
        #     if node.val <= previous:
        #         return False
        #     previous = node.val
        #     root = node.right       # 往往第一pop出来的是叶节点，没有子节点了，下一次pop就会是其父节点，就可能会有右节点了
        # return True

    def numTrees(self, n: int) -> int:
        """
        96.不同的二叉搜索树
        中等 2023.03.20
        :param n:
        :return:
        """
        dp = [0] * (n + 1)
        dp[0] = dp[1] = 1
        # 分别用连续的i个数构建BST的类别数
        for i in range(2, n + 1):  # 假设此时一共有i个连续的数
            for j in range(i - 1 + 1):
                dp[i] += dp[j] * dp[i - 1 - j]
        return dp[n]

    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        """
        94.二叉树的中序遍历
        简单 2023.03.22
        题解：你猜这简单么？
        :param root:
        :return:
        """
        if not root:
            return []
        stack = []  # 中序遍历用栈还是队列？栈！
        res = []
        while stack or root:
            while root:
                stack.append(root)
                root = root.left
            node = stack.pop()
            res.append(node.val)
            root = node.right
        return res

    def removeDuplicates(self, nums: List[int]) -> int:
        """
        80.删除有序数组中的重复项II
        中等 2023.03.22
        题解：还是比较难的
        :param nums:
        :return:
        """

        # 还是得看答案的代码呀
        def keep(k):
            curIdx = 0
            for n in nums:
                if curIdx < k or n != nums[curIdx - k]:
                    nums[curIdx] = n
                    curIdx += 1
            return curIdx

        return keep(2)

    def subsets(self, nums: List[int]) -> List[List[int]]:
        """
        78.子集
        中等 2023.03.23
        题解：回溯 不太会呀
        :param nums:
        :return:
        """
        # 重写一遍
        res = []
        path = []

        def fun(nums, startInd):
            res.append(path[:])
            for i in range(startInd, len(nums)):
                path.append(nums[i])
                fun(nums, i + 1)
                path.pop()

        fun(nums, 0)
        return res

        # res = []
        # path = []
        #
        # def fun(nums, startInd):
        #     res.append(path[:])
        #     for i in range(startInd, len(nums)):
        #         path.append(nums[i])
        #         fun(nums, i + 1)
        #         path.pop()
        #
        # fun(nums, 0)
        # return res

    def combine(self, n: int, k: int) -> List[List[int]]:
        """
        77.组合
        中等 2023.03.25
        题解：
        :param n:
        :param k:
        :return:
        """
        res, path = [], []

        def fun(startInd):
            if len(path) == k:
                res.append(path[:])
                return
            for i in range(startInd, n + 1):
                path.append(i)
                fun(i + 1)
                path.pop()

        fun(1)
        return res

    def sortColors(self, nums: List[int]) -> None:
        """
        75.颜色分类
        中等 2023.03.29
        题解：in place 冒泡
        :param nums:
        :return:
        """
        #冒泡，往前冒
        for i in range(len(nums) - 1):
            for j in range(len(nums) - 1, i, -1):
                if nums[j - 1] > nums[j]:
                    nums[j - 1], nums[j] = nums[j], nums[j - 1]

        # 冒泡，往后冒
        # for i in range(len(nums) - 1):      # 冒泡多少次
        #     for j in range(1, len(nums) - i):   # 每次冒泡从哪开始，我可以选择往后冒或往前冒
        #         if nums[j - 1] > nums[j]:
        #             nums[j - 1], nums[j] = nums[j], nums[j - 1]

    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        """
        74.搜索二维矩阵
        中等 2023.03.29
        题解：可尝试定位行的时候也是用二分法
        :param matrix:
        :param target:
        :return:
        """
        # 解法二：使用二分法. 特别注意，第一次比较行首元素，第二次均可
        max_value_row = [row[0] for row in matrix]
        left, right = 0, len(max_value_row) - 1
        while left <= right:
            mid = (left + right) // 2
            if max_value_row[mid] < target:
                left = mid + 1
            elif max_value_row[mid] > target:
                right = mid - 1
            else:
                return True
        # print(matrix[right])     # 定位到left行
        # exit()
        cur_row_values = matrix[right]
        left, right = 0, len(cur_row_values) - 1
        while left <= right:
            mid = (left + right) // 2
            if cur_row_values[mid] < target:
                left = mid + 1
            elif cur_row_values[mid] > target:
                right = mid - 1
            else:
                return True
        return False

        # 解法一：最简单的思路
        # # 定位行
        # cur_row = -1
        # for i, line in enumerate(matrix):
        #     if target <= line[-1]:
        #         cur_row = i
        #         break       # 切记，找到之后break！
        # if cur_row == -1:
        #     return False
        # # 定位列，二分？
        # for i, n in enumerate(matrix[cur_row]):
        #     if n == target:
        #         return True
        # return False

    def climbStairs(self, n: int) -> int:
        """
        70.爬楼梯
        简单 2023.03.30
        题解：费锲那波数列
        :param n:
        :return:
        """
        f = [None] * (n + 1)
        f[0] = 1
        f[1] = 1
        for i in range(2, n + 1):
            f[i] = f[i - 1] + f[i - 2]
        return f[n]

    def mySqrt(self, x: int) -> int:
        """
        69.x的平方根
        简单 2023.03.30
        题解：牛顿迭代法
        :param x:
        :return:
        """
        n = 2   # 初始化为任意值
        iters = 20
        while iters:
            iters -= 1
            n = (n + x / n) / 2
        return int(n)

    def minPathSum(self, grid: List[List[int]]) -> int:
        """
        64.最小路径和
        中等 2023.03.30
        题解：所谓不谋而合
        :param grid:
        :return:
        """
        rows, cols = len(grid), len(grid[0])
        dp = [[-1] * cols for _ in range(rows)]
        dp[0][0] = grid[0][0]
        for i in range(rows):
            for j in range(cols):
                if i - 1 >= 0 and j - 1 >= 0:   # 当前位置可从 左、上  方向到达
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]
                elif i - 1 < 0 <= j - 1:    # 当前位置可从 左 方向到达
                    dp[i][j] = dp[i][j - 1] + grid[i][j]
                elif j - 1 < 0 <= i - 1:    # 当前位置可从 上 方向到达
                    dp[i][j] = dp[i - 1][j] + grid[i][j]
        return dp[-1][-1]

    def uniquePaths(self, m: int, n: int) -> int:
        """
        62.不同路径
        中等 2023.04.01
        题解：
        :param m: 行
        :param n: 列
        :return:
        """
        dp = [[1] * n] + [[1] + [0] * (n - 1) for _ in range(m - 1)]
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[-1][-1]

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        """
        56.合并区间
        中等 2023.04.01
        题解：自己想的太复杂了，直接看题解吧
        :param intervals:
        :return:
        """
        intervals.sort(key=lambda x: x[0])
        res = []
        for inter in intervals:
            if not res or res[-1][-1] < inter[0]:   # 什么情况直接放进res：res为空 或 不重合
                res.append(inter)
            elif res[-1][-1] >= inter[0]:
                res[-1][-1] = max(res[-1][-1], inter[-1])
        return res

    def canJump(self, nums: List[int]) -> bool:
        """
        55.跳跃游戏
        中等 2023.04.03
        题解：
        :param nums:
        :return:
        """
        max_steps = 0
        for i, n in enumerate(nums):
            if i <= max_steps < i + n:  # i <= max_steps表示能跳到i位置；i + n表示从位置i能到达的最远位置，max_steps < i + n表示尚不能到达从位置i起始的最远位置——更新max_steps
                max_steps = i + n   # 能到达i了，更新后就能到达位置i + n了
        return max_steps >= len(nums) - 1




if __name__ == '__main__':
    sl = Solution()

    nums = [3, 2, 1, 0, 4]
    print(sl.canJump(nums))

    # intervals = [[1, 4], [4, 5]]
    # print(sl.merge(intervals))

    # print(sl.uniquePaths(3, 2))

    # grid = [[1,2,3],[4,5,6]]
    # print(sl.minPathSum(grid))

    # print(sl.mySqrt(9))

    # print(sl.climbStairs(3))


    # matrix = [[1]]
    # target = 3
    # print(sl.searchMatrix(matrix, target))

    # nums = [2, 0, 2, 1, 1, 0]
    # sl.sortColors(nums)
    # print(nums)

    # for i in range(5, 1, -1):
    #     print(i)

    # print(sl.combine(4, 2))

    # nums = [1, 2, 3]
    # print(sl.subsets(nums))
