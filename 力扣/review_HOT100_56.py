"""
2023.04.05 从第56题开始复习，只看简单、中等
"""
from typing import List, Optional


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def canJump(self, nums: List[int]) -> bool:
        """
        55.跳跃游戏
        2023.04.05 中等
        题解：好像容易解，其实挺绕的，要细心、冷静
        :param nums:
        :return:
        """
        max_steps = 0
        for i, n in enumerate(nums):
            if i <= max_steps < i + n:
                max_steps = i + n
        return max_steps >= len(nums) - 1

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        """
        56.合并区间
        2023.04.05 中等
        题解：思路第一时间还是没有想起来。
        :param intervals:
        :return:
        """
        intervals.sort(key=lambda x: x[0])
        res = []
        for inter in intervals:
            if not res or res[-1][-1] < inter[0]:
                res.append(inter)
            elif res[-1][-1] >= inter[0]:  # 这个判断其实都不用写
                res[-1][-1] = max(res[-1][-1], inter[1])
        return res

    def uniquePaths(self, m: int, n: int) -> int:
        """
        62.不同路径
        中等 2023.04.11
        题解：只能向下和向右走，则到达首行首列元素路径只有1，初始化首行首列为1，其余为0。到达其他元素路径数为左边+上边
        :param m:
        :param n:
        :return:
        """
        res = [[1] * n] + [[1] + [0] * (n - 1) for _ in range(m - 1)]
        for i in range(1, m):
            for j in range(1, n):
                res[i][j] = res[i - 1][j] + res[i][j - 1]
        return res[-1][-1]

    def minPathSum(self, grid: List[List[int]]) -> int:
        """
        64.最小路径和
        中等 2023.04.11
        题解：终于体会到动态规划的意思了
        :param grid:
        :return:
        """
        m = len(grid)
        n = len(grid[0])
        for i in range(m):
            for j in range(n):
                if i - 1 < 0 and j - 1 < 0:
                    grid[i][j] = grid[i][j]
                elif i - 1 < 0 <= j - 1:  # 处理首行元素
                    grid[i][j] += grid[i][j - 1]
                elif i - 1 >= 0 > j - 1:  # 处理首列元素
                    grid[i][j] += grid[i - 1][j]
                else:
                    grid[i][j] += min(grid[i - 1][j], grid[i][j - 1])
        return grid[-1][-1]

    def climbStairs(self, n: int) -> int:
        """
        70.爬楼梯
        简单 2023.04.13
        题解：似乎是动态规划？
        :param n:
        :return:
        """
        f = [-1] * n
        f[0] = 1
        if n == 1:
            return f[-1]
        f[1] = 2
        for i in range(2, n):
            f[i] = f[i - 1] + f[i - 2]
        return f[-1]

    def sortColors(self, nums: List[int]) -> None:
        """
        75.颜色分类
        中等 2023.04.13
        题解：看了题解，冒泡
            方法二 https://leetcode.cn/problems/sort-colors/solution/yan-se-fen-lei-by-leetcode-solution/
        :param nums:
        :return:
        """
        p0 = p1 = 0  # 2个指针初始化指向索引0
        for i in range(len(nums)):
            if nums[i] == 0:
                nums[i], nums[p0] = nums[p0], nums[i]
                if p0 < p1:  # 因为p0通常在p1左面嘛，有可能在与位置i置换时把元素1给换跑[换跑就是换到位置i了嘛]，因此要换回来
                    nums[i], nums[p1] = nums[p1], nums[i]
                p0 += 1
                p1 += 1
            elif nums[i] == 1:
                nums[i], nums[p1] = nums[p1], nums[i]
                p1 += 1  # p1通常都在p0右面嘛，所以p1这里处理较简单，p0那里较复杂
        return nums

        # 冒泡 从后往前冒
        # for i in range(len(nums) - 1):
        #     for j in range(len(nums) - 1, i, -1):
        #         if nums[j - 1] > nums[j]:
        #             nums[j - 1], nums[j] = nums[j], nums[j - 1]
        # return nums

        # 冒泡 从前往后冒
        # for i in range(len(nums) - 1):
        #     for j in range(0, len(nums) - i - 1):
        #         if nums[j] > nums[j + 1]:
        #             nums[j], nums[j + 1] = nums[j + 1], nums[j]
        # return nums

    def subsets(self, nums: List[int]) -> List[List[int]]:
        """
        78.子集
        中等 2023.04.18
        题解：看了之前题解
        :param nums:
        :return:
        """
        path = []  # 存放当前子集
        res = []  # 存放所有子集

        def fun(startInd):
            res.append(path[:])
            for i in range(startInd, len(nums)):  # 之所以从startInd开始，因为题目不要求结果顺序，所以是不放回
                path.append(nums[i])
                fun(i + 1)
                path.pop()

        fun(0)
        return res

    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        """
        94.二叉树的中序遍历
        中等 2023.04.18
        题解：递归适用于前序、中序、后序
            迭代好理解，但不同顺序遍历代码不同 https://leetcode.cn/problems/binary-tree-inorder-traversal/solution/dong-hua-yan-shi-94-er-cha-shu-de-zhong-xu-bian-li/
        :param root:
        :return:
        """
        # 方法一：递归实现，很容易应用到前序、中序、后序遍历
        # res = []
        #
        # def fun(root):
        #     if not root:    # 遍历到空 则返回
        #         return
        #     # 更改下面三个语句顺序即可完成 前序、中序、后序遍历
        #     fun(root.left)
        #     res.append(root.val)
        #     fun(root.right)
        # fun(root)
        # return res

        # 方法二：迭代实现
        res = []
        stack = []  # 迭代实现需要 栈
        while stack or root:  # 不能只考虑初次执行while
            if root:  # 一直往左孩子走，直到走到最底层的左孩子
                stack.append(root)
                root = root.left
            else:  # root为空，此时从stack往外pop
                node = stack.pop()
                res.append(node.val)
                root = node.right  # 还得遍历右孩子呢
        return res

    def numTrees(self, n: int) -> int:
        """
        96.不同的二叉搜索树
        中等 2023.04.20
        :param n:
        :return:
        """
        dp = [0 for _ in range(n + 1)]
        dp[0] = 1
        dp[1] = 1
        for i in range(2, n + 1):  # i个节点可组成多少种BST。注意，参数n是有几个节点，所以range里面取到n + 1
            for j in range(0, i):  # 共i个节点，1个根节点，则左右子树共i-1个节点。j的取值为0~i-1，range要写到i呀！！！
                dp[i] += dp[j] * dp[i - 1 - j]  # 左右子树节点数之和==i-1
        return dp[-1]

    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        """
        98.验证二叉搜索树
        中等 2023.04.20
        题解：中序遍历依次增大，否则不是BST
        :param root:
        :return:
        """
        # 方法一：递归
        # res = []
        #
        # def fun(root):
        #     if not root:
        #         return
        #     fun(root.left)
        #     res.append(root.val)
        #     fun(root.right)
        # fun(root)
        #
        # for i in range(1, len(res)):
        #     if res[i - 1] >= res[i]:
        #         return False
        # return True

        # 方法二：迭代
        # 94题的迭代是不考虑空元素的，本题存在空元素;而98题的node为空时进不去stack 这是关键
        #                                           94题迭代法在本题不通，就不能使用保留中序遍历的思路，而是要保留上一个中序遍历结果即可
        stack = []
        inorder = float('-inf')  # 先初始化一个很小的数，中序遍历第一个数的时候是没有前一个数的
        while root or stack:  # 二者有一个不空就能继续，要么root还有元素，要么stack还没遍历完
            while root:
                stack.append(root)
                root = root.left
            node = stack.pop()
            if node.val <= inorder:
                return False
            inorder = node.val
            root = node.right
        return True

    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        """
        101.对称二叉树
        简单 2023.04.22
        题解：我觉得中序遍历就能判断  注：不是判断搜索二叉树
        :param root:
        :return:
        """
        if not root or not (root.left or root.right):  # 空树、root无孩子
            return True
        queue = [root.left, root.right]  # 似乎不一定按队列顺序
        while queue:
            # 判断取出的2个节点
            node1 = queue.pop()
            node2 = queue.pop()
            if not (node1 or node2):  # 两个均为空节点
                continue
            if not (node1 and node2):  # 其中1个为空
                return False
            if node1.val != node2.val:
                return False
            queue.extend(node1.left, node2.right)
            queue.extend(node1.right, node2.left)

        return True

    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        """
        102.二叉树的层序遍历
        中等 2023.04.22
        题解：队列
        :param root:
        :return:
        """
        # 题目要求 逐层
        if not root:
            return []
        queue = [root]
        res = []
        while queue:  # 以层为单位处理
            num_layer_node = len(queue)
            tmp = []
            for _ in range(num_layer_node):
                node = queue.pop(0)
                if node:
                    tmp.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(tmp)
        return res

        # 仅实现了层序遍历，返回结果没有包含 逐层
        # if not root:
        #     return True
        # queue = [root]
        # res = []
        # while queue:
        #     node = queue.pop(0)
        #     if node:
        #         res.append(node.val)
        #         queue.extend([node.left, node.right])     # 这是BUG，只判断node空否，其子节点并没有判断就入队列了
        # return res

    def maxDepth(self, root: Optional[TreeNode]) -> int:
        """
        104.二叉树的最大深度
        简单 2023.04.23
        题解：层序遍历，返回深度
        :param root:
        :return:
        """
        if not root:
            return 0
        queue = [root]
        seq_travel = []
        while queue:
            num_layer_nodes = len(queue)
            tmp = []    # 每层的节点
            for _ in range(num_layer_nodes):
                node = queue.pop(0)
                if node:
                    tmp.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            seq_travel.append(tmp)
        return len(seq_travel)

    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        """
        105.从前序与中序遍历序列构造二叉树
        中等 2023.04.23
        题解：直接看答案，递归不好理解
        :param preorder:
        :param inorder:
        :return:
        """
        # 迭代更麻烦，再写一遍递归吧
        inorder_val2ind = {val: i for i, val in enumerate(inorder)}

        def myBuildTree(preorder_l, preorder_r, inorder_l, inorder_r):
            """myBuildTree处理1棵子树
            参数指示的同一[子]树的前序、中序遍历的开始结束索引"""
            # myBuildTree处理的子树也有根节点，其在先序、中序遍历位置
            if preorder_l > preorder_r:
                return None
            preorder_rootInd = preorder_l
            inorder_rootInd = inorder_val2ind[preorder[preorder_l]]
            subTree_len = inorder_rootInd - inorder_l   # myBuildTree所处理子树的长度
            root = TreeNode(preorder[preorder_rootInd])     # 建立根节点
            # 建立root.left时，要想到myBuildTree的所有参数均指示[同一子树]。如何使用myBuildTree的参数来表示root的左子树 所在 [同一子树] 的先序、中序遍历索引
            root.left = myBuildTree(preorder_rootInd + 1, preorder_rootInd + subTree_len, inorder_l, inorder_rootInd - 1)
            # 原理同上
            root.right = myBuildTree(preorder_rootInd + 1 + subTree_len, preorder_r, inorder_rootInd + 1, inorder_r)
            return root

        n = len(preorder)
        return myBuildTree(0, n - 1, 0, n - 1)

        # 递归 不推荐 不好懂呀
        # inorder_val2ind = {val: i for i, val in enumerate(inorder)}
        #
        # def myBuildTree(preorder_left, preorder_right, inorder_left, inorder_right):
        #     if preorder_left > preorder_right:
        #         return None
        #     preorder_rootInd = preorder_left
        #     inorder_rootInd = inorder_val2ind[preorder[preorder_rootInd]]
        #     subTree_len = inorder_rootInd - inorder_left
        #     root = TreeNode(preorder[preorder_rootInd])
        #     root.left = myBuildTree(preorder_left + 1, preorder_left + subTree_len, inorder_left, inorder_rootInd - 1)
        #     root.right = myBuildTree(preorder_left + subTree_len + 1, preorder_right, inorder_rootInd + 1, inorder_right)
        #
        #     return root
        #
        # tree_len = len(preorder)
        # root = myBuildTree(0, tree_len - 1, 0, tree_len - 1)
        # return root

    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        114.二叉树展开为链表
        中等 2023.04.24
        题解：方法一 https://leetcode.cn/problems/flatten-binary-tree-to-linked-list/solution/xiang-xi-tong-su-de-si-lu-fen-xi-duo-jie-fa-by--26/
        展开后为先序遍历-->先序遍历 不行 还是看解析吧
        :param root:
        :return:
        """
        while root:
            if not root.left:   # 如果root的左孩子空，就处理右孩子
                root = root.right
            else:   # root的左孩子不空：则先找左孩子的最右，然后把root的右孩子挂上，再然后把root的左孩子覆盖root的右孩子位置
                tmp_root = root.left
                while tmp_root.right:
                    tmp_root = tmp_root.right
                tmp_root.right = root.right
                root.right = root.left
                root.left = None
                root = root.right

    def maxProfit(self, prices: List[int]) -> int:
        """
        121.买卖股票的最佳时机
        简单 2023.04.25
        题解：方法二 一次遍历 https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/solution/121-mai-mai-gu-piao-de-zui-jia-shi-ji-by-leetcode-/
            循环是愚蠢的
        :param prices:
        :return:
        """
        # 一次遍历
        min_price = prices[0]
        profit = 0
        for price in prices:
            profit = max(profit, price - min_price)
            min_price = min(min_price, price)
        return profit


        # 超时 此思路可能源于Python语言循环实现
        # res = float('-inf')
        # flag = False
        # for i in range(len(prices) - 1, 0, -1):
        #     for j in range(i):
        #         if prices[i] > prices[j] and prices[i] - prices[j] > res:
        #             res = prices[i] - prices[j]
        #             flag = True
        # return res if flag else 0

    def longestConsecutive(self, nums: List[int]) -> int:
        """
        128.最长连续序列
        中等 2023.04.25
        题解：直接看答案代码更好理解 https://leetcode.cn/problems/longest-consecutive-sequence/solution/zui-chang-lian-xu-xu-lie-by-leetcode-solution/
        :param nums:
        :return:
        """
        nums_set = set(nums)
        res = 0
        for num in nums_set:
            if num - 1 not in nums_set:     # 只有当前num是连续数字的第一个才继续
                cur_len = 0
                while num in nums_set:
                    num += 1
                    cur_len += 1
                res = max(res, cur_len)
        return res

    def singleNumber(self, nums: List[int]) -> int:
        """
        136.只出现一次的数字
        简单 2023.04.26
        题解：res = 1 ^ 2 ^ 3 ^ 4 ^ 5 ^ 4 ^ 3 ^ 1 ^ 2 结果5
            最后解法 https://leetcode.cn/problems/single-number/solution/xue-suan-fa-jie-guo-xiang-dui-yu-guo-cheng-bu-na-y/
        :param nums:
        :return:
        """
        res = nums[0]
        for n in nums[1:]:
            res = res ^ n
        return res

    def singleNumber_3(self, nums: List[int]) -> int:
        """
        137.只出现一次的数字II
        中等 2023.04.26
        题解：二进制
        :param nums:
        :return:
        """
        # 题目要求的取值范围刚好可用32位二进制表示，因为Python不区分正负数，所以在处理i=31时，如果模3不为0的话，
        # ans不能执行ans |= (1 << i)，这将使得ans超过2 ** 31大小，不符题意了。而C++等区分正负数的语言则没有这个问题。
        ans = 0
        for i in range(32):
            total = sum([(n >> i) & 1 for n in nums])
            if total % 3:
                if i == 31:
                    ans -= (1 << 31)
                else:
                    ans |= (1 << i)
        return ans

    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        """
        139.单词拆分
        中等 2023.04.28
        题解：方法一 https://leetcode.cn/problems/word-break/solution/dong-tai-gui-hua-ji-yi-hua-hui-su-zhu-xing-jie-shi/
        :param s:
        :param wordDict:
        :return:
        """
        n = len(s)
        dp = [False] * (n + 1)  # 表示s的前0~n个字符组成的单词，是否在字典中
        dp[0] = True    # 前0个字符组成的单词是空的，我们认为是可以表示的(不然后面if条件走不通了)
        for i in range(n):  # 注意两个for循环的边界
            for j in range(i + 1, n + 1):
                if dp[i] and s[i:j] in wordDict:    # 这是关键，只有前i位可以组成的单词在字典中，前j位(即if表达的含义)才有可能在字典中；不能仅判断s[i:j]是否在字典中，这样判断的即便在字典中，也不符合dp的意义
                    dp[j] = True    # 注意，dp[j]表示前j个字符，不包括索引j
        return dp[-1]

    def hasCycle(self, head: Optional[ListNode]) -> bool:
        """
        141.环形链表
        简单 2023.04.28
        题解：竟然没做出来哈哈
            方法一 哈希表(set实现的) https://leetcode.cn/problems/linked-list-cycle/solution/huan-xing-lian-biao-by-leetcode-solution/
        :param head:
        :return:
        """
        seen = set()
        while head:
            if head in seen:
                return True
            seen.add(head)
            head = head.next
        return False

    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        142.环形链表II
        中等 2023.05.05
        题解：
        :param head:
        :return:
        """
        seen = set()
        while head:
            if head in seen:
                return head
            seen.add(head)
            head = head.next
        return None

    def reorderList(self, head: Optional[ListNode]) -> None:
        """
        143.重排链表
        中等 2023.05.05
        题解：用stack把所有节点保存下来，后面看答案吧哈哈
            方法1 https://leetcode.cn/problems/reorder-list/solution/zhong-pai-lian-biao-by-leetcode-solution/
        Do not return anything, modify head in-place instead.
        """
        stack = []
        tmp = head
        while tmp:
            stack.append(tmp)
            tmp = tmp.next
        i, j = 0, len(stack) - 1
        while i < j:
            stack[j].next = stack[i].next
            stack[i].next = stack[j]
            i += 1
            j -= 1
            if i == j:
                break
        stack[i].next = None    # 注意处理最后一个节点！

    # from collections import OrderedDict     # 方法一
    # class DLinkedNode:      # 方法二
    #     def __init__(self, key=0, val=0):
    #         self.key = key
    #         self.val = val
    #         self.prev = None
    #         self.next = None
    #
    # class LRUCache:
    #     """
    #     146.LRU缓存
    #     中等 2023.05.06
    #     题解：首先想到OrderedDict 居然对了，阿弥陀佛！
    #         官方答案 https://leetcode.cn/problems/lru-cache/solution/lruhuan-cun-ji-zhi-by-leetcode-solution/
    #     """
    #     # 方法二：哈希表+双向链表
    #     def __init__(self, capacity: int):
    #         self.capacity = capacity
    #         self.cache = dict()     # 哈希表的作用
    #         self.head = DLinkedNode()   # 伪头结点
    #         self.tail = DLinkedNode()   # 伪尾结点
    #         self.head.next = self.tail
    #         self.tail.prev = self.head
    #
    #     def get(self, key: int) -> int:
    #         if key not in self.cache:
    #             return -1
    #         node = self.cache[key]
    #         self.movetoHead(node)
    #         return node.val
    #
    #     def put(self, key: int, value: int) -> None:
    #         if key not in self.cache:
    #             node = DLinkedNode(key, value)
    #             self.cache[key] = node
    #             self.addToHead(node)
    #             if len(self.cache) > self.capacity:
    #                 node = self.removeTail()   # 移除最后一个节点
    #                 self.cache.pop(node.key)
    #         else:
    #             self.cache[key].val = value
    #             self.movetoHead(self.cache[key])
    #
    #     # 下面这些小方法，只修改双向链表，不修改self.cache
    #     def removeNode(self, node: DLinkedNode):
    #         node.prev.next = node.next
    #         node.next.prev = node.prev
    #
    #     def movetoHead(self, node: DLinkedNode):
    #         """将node移到第一个节点，分2步：1.removeNode 2.addToHead"""
    #         self.removeNode(node)
    #         self.addToHead(node)
    #
    #     def addToHead(self, node: DLinkedNode):
    #         """将node放到第一个节点位置"""
    #         node.prev = self.head
    #         node.next = self.head.next
    #         self.head.next.prev = node
    #         self.head.next = node
    #
    #     def removeTail(self):
    #         """移除最后的node，不需要参数"""
    #         node = self.tail.prev   # 最后一个节点
    #         self.removeNode(node)
    #         return node
    #
    #     # 方法一：Python内置OrderedDict，但不推荐
    #     # def __init__(self, capacity: int):
    #     #     self.max_len = capacity
    #     #     self.d = OrderedDict()
    #     #
    #     # def get(self, key: int) -> int:
    #     #     if key in self.d.keys():
    #     #         v = self.d.get(key)
    #     #         self.d.move_to_end(key, last=False)     # move_to_end()是必要的. popitem()的LIFO的进入顺序应该就是指的当前OrderedDict元素顺序
    #     #         return v
    #     #     return -1
    #     #
    #     # def put(self, key: int, value: int) -> None:
    #     #     # 要注意添加的逻辑，是否超出最大长度
    #     #     self.d[key] = value
    #     #     self.d.move_to_end(key, last=False)
    #     #     if self.d.__len__() > self.max_len:
    #     #         _ = self.d.popitem(last=True)

    def maxProduct(self, nums: List[int]) -> int:
        """
        152.乘积最大子数组
        2023.05.08 中等
        题解：首先想到类似动态规划
            https://leetcode.cn/problems/maximum-product-subarray/solution/hua-jie-suan-fa-152-cheng-ji-zui-da-zi-xu-lie-by-g/
            也可参考官方答案
        :param nums:
        :return:
        """
        imax = imin = 1     # 因为从索引0开始遍历，用到乘，所以初始化为1
        max_f = float('-inf')   # 记录最大值，所以初始化为最小值
        for n in nums:
            if n < 0:
                imax, imin = imin, imax
            imax = max(n, n * imax)
            imin = min(n, n * imin)
            max_f = max(max_f, imax)
        return max_f

    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        148.排序链表
        2023.05.11 中等
        题解：难呀 方法一 https://leetcode.cn/problems/sort-list/solution/pai-xu-lian-biao-by-leetcode-solution/
        :param head:
        :return:
        """
        def sortFunc(head: ListNode, tail: ListNode):
            # head tail指向同一子链表首尾，更确切地说是两个自链表的开头，即head指示的链表不包括节点tail
            if not head:
                return head
            # 这里为什么要判断head.next == tail，是为了处理tail不空的时候，此时head,tail指示的是两个子链表序列的开头(看递归调用sortFunc就明白了)，
            # 都不空。现在head.next==tail，就是head指示的自链表序列只有1个节点。大意如此，仔细体会吧
            if head.next == tail:   # 子链表就1个节点
                head.next = None
                return head

            slow = fast = head
            while fast != tail:
                slow = slow.next
                fast = fast.next
                if fast != tail:
                    fast = fast.next
            mid = slow
            # head_1 = sortFunc(head, mid)
            # head_2 = sortFunc(mid, tail)
            # return merge(head_1, head_2)
            return merge(sortFunc(head, mid), sortFunc(mid, tail))

        def merge(head_1: ListNode, head_2: ListNode):
            # 这里也将head_1、head_2作为两个独立的链表考虑，方便实现; 比如while判断时tmp1和tmp2空不空时，并没有考虑二者之间的关系，二者之间也可能没关系
            dummyHead = ListNode(0)     # 伪头结点
            tmp, tmp1, tmp2 = dummyHead, head_1, head_2
            while tmp1 and tmp2:
                if tmp1.val <= tmp2.val:
                    tmp.next = tmp1
                    tmp1 = tmp1.next
                else:
                    tmp.next = tmp2
                    tmp2 = tmp2.next
                tmp = tmp.next
            # 在做最后的拼接时，一定要判断tmp1、tmp2是否为空
            # 若直接tmp.next = tmp1, tmp.next = tmp2，看起来人畜无害，若tmp1不空但tmp2空，则发生覆盖了，后面的断了
            if tmp1:
                tmp.next = tmp1
            elif tmp2:
                tmp.next = tmp2
            return dummyHead.next

        return sortFunc(head, None)     # head, None可理解为一个子链表的首位，或两个子链表的开头

    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        """
        160.相交链表
        2023.05.14 简单
        题解：首先想到暴力，但效率很低
            方法二：双指针 https://leetcode.cn/problems/intersection-of-two-linked-lists/solution/xiang-jiao-lian-biao-by-leetcode-solutio-a8jn/
        :param headA:
        :param headB:
        :return:
        """
        # 用时、内存消耗要低很多
        if not (headA and headB):   # 二者任何一个为空，则进入
            return None
        pA, pB = headA, headB
        while pA != pB:
            pA = headB if pA is None else pA.next
            pB = headA if pB is None else pB.next
        return pA
        # 暴力，效率较低
        res = []
        while headA:
            res.append(headA)
            headA = headA.next
        while headB:
            if headB in res:
                return headB
            headB = headB.next
        return None

    def majorityElement(self, nums: List[int]) -> int:
        """
        169.多数元素
        2023.05.14 简单
        题解：答案给了一个提示，出现次数⌊ n/2 ⌋的元素即众数，先排序，返回索引⌊ n/2 ⌋的元素即为答案

        :param nums:
        :return:
        """
        # 使用冒泡排序 超时
        # for i in range(len(nums) - 1):
        #     for j in range(len(nums) - 1 - i):
        #         if nums[j] > nums[j + 1]:
        #             nums[j], nums[j + 1] = nums[j + 1], nums[j]

        # 还是得复习排序算法
        nums.sort()
        print(f'排序结果: {nums}')
        return nums[len(nums) // 2]

    def rob(self, nums: List[int]) -> int:
        """
        198.打家劫舍
        2023.05.16 中等
        题解：这个答案没看太懂 https://leetcode.cn/problems/house-robber/solution/dong-tai-gui-hua-jie-ti-si-bu-zou-xiang-jie-cjavap/
            答案评论：因为dp[k]代表的意义是偷前k间房子得到的最大金额，那么你偷了当前房子，就需要偷前k - 2间房子的最大金额；
                你不偷当前房子，就需要前k-1间房子的最大金额。
                至于这些金额是怎么算出来的我们并不关心，dp数组的定义可以确保我们拿到的就是偷前k间房的最大金额。
            答案评论：这个解法的含义就是，你当前位于K的房子，你可以选择：1、偷 2、不偷。
                第一种方案不能偷k-1的房子，所以f(k)=H(k-1)+f(k-2)；
                第二种方案可以偷前(k-1)个房子中的任意房子，所以f(k)=f(k-1)；
                最终小偷在两种方案中做权衡，f(k)=max(H(k-1)+f(k-2),f(k-1))
        :param nums:
        :return:
        """
        # 注意nums索引与dp索引
        n = len(nums)
        dp = [0] * (n + 1)      # 索引表示偷 前索引间房 所获得的最大金额
        dp[0] = 0
        dp[1] = nums[0]
        for k in range(2, n + 1):   # dp索引对应[2, n]
            dp[k] = max(dp[k - 2] + nums[k - 1], dp[k - 1])       # 偷 or 不偷  索引第k间房
        return dp[n]

    def numIslands(self, grid: List[List[str]]) -> int:
        """
        200.岛屿数量
        2023.05.16 中等
        题解：深度优先遍历 https://leetcode.cn/problems/number-of-islands/solution/dao-yu-shu-liang-by-leetcode/
        :param grid:
        :return:
        """
        def dfs(grid, x, y):
            grid[x][y] = 0
            nr = len(grid)
            nc = len(grid[0])
            for x, y in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= x < nr and 0 <= y < nc and grid[x][y] == "1":
                    dfs(grid, x, y)

        nr = len(grid)
        if nr == 0:
            return 0
        nc = len(grid[0])
        num_islands = 0
        for x in range(nr):
            for y in range(nc):
                if grid[x][y] == "1":
                    num_islands += 1
                    dfs(grid, x, y)
        return num_islands

    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        206.反转链表
        2023.05.17 简单
        题解：暴力通不过测试
            方法一 迭代https://leetcode.cn/problems/reverse-linked-list/solution/fan-zhuan-lian-biao-by-leetcode-solution-d1k2/
        :param head:
        :return:
        """
        prev = None     # 往回指的时候需要引用前一个节点
        curr = head
        while curr:
            next = curr.next    # 往回指之前先保存后一个节点，不然没法迭代了
            curr.next = prev    # 往回指
            prev = curr     # prev后移
            curr = next     # curr后移
        return prev     # 注意，这里不是返回head，head没用了

class MinStack:
    """
    155.最小栈
    2023.05.13 中等
    题解：
    """

    def __init__(self):
        self.stack = []
        self.minstack = []

    def push(self, val: int) -> None:
        if not len(self.stack):
            self.stack.append(val)
            self.minstack.append(val)
        else:
            self.stack.append(val)
            self.minstack.append(min(val, self.minstack[-1]))

    def pop(self) -> None:
        self.stack.pop()
        self.minstack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.minstack[-1]


if __name__ == '__main__':
    sl = Solution()

    nums = [1]
    print(sl.rob(nums))

    # nums = [2,-1,1,1]
    # print(sl.maxProduct(nums))

    # s = "catsandog"
    # wordDict = ["cats", "dog", "sand", "and", "cat"]
    # print(sl.wordBreak(s, wordDict))
    # nums = [-2,-2,1,1,4,1,4,4,-4,-2]
    # print(sl.singleNumber_3(nums))

    # nums = []
    # print(sl.longestConsecutive(nums))

    # prices = [2, 4, 1]
    # print(sl.maxProfit(prices))

    # print(sl.numTrees(1))

    # nums = [1, 2, 3]
    # print(sl.subsets(nums))

    # nums = [2,0,2,1,1,0]
    # print(sl.sortColors(nums))

    # print(sl.climbStairs(1))

    # grid = [[1,2,3],[4,5,6]]
    # print(sl.minPathSum(grid))

    # print(sl.uniquePaths(3, 2))

    # intervals = [[1,4],[4,5]]
    # print(sl.merge(intervals))
