from typing import Optional, List


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children


class Solution:
    """ ------------------- 深度优先遍历 递归法  ----------------------"""
    """ 前序遍历 递归实现，稍加修改得到 中序遍历、后序遍历 """
    def preorderTraversal(self, root:TreeNode):
        """ 导读：
            前序遍历 递归 """
        res = []

        def backtracking(node):
            # 终止条件
            if not node:
                return
            # 单层递归逻辑
            res.append(node.val)
            backtracking(node.left)
            backtracking(node.right)

        backtracking(root)
        return res

    def inorderTraversal(self, root:TreeNode):
        """ 导读：
            中序遍历 递归 """
        res = []

        def backtracking(node):
            # 终止条件
            if not node:
                return
            # 单层递归逻辑
            backtracking(node.left)
            res.append(node.val)
            backtracking(node.right)

        backtracking(root)
        return res

    def postorderTraversal(self, root:TreeNode):
        """ 导读：
            后序遍历 递归 """
        res = []

        def backtracking(node):
            # 终止条件
            if not node:
                return
            # 单层递归逻辑
            backtracking(node.left)
            backtracking(node.right)
            res.append(node.val)

        backtracking(root)
        return res

    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        """ 144.二叉树的前序遍历 """
        res = []

        def backtracking(node):
            # 终止条件
            if not node:
                return
            # 单层递归逻辑
            res.append(node.val)
            backtracking(node.left)
            backtracking(node.right)

        backtracking(root)
        return res

    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        """ 145.二叉树的后序遍历 """
        res = []

        def backtracking(node):
            # 终止条件
            if not node:
                return
            # 单层递归逻辑
            backtracking(node.left)
            backtracking(node.right)
            res.append(node.val)

        backtracking(root)
        return res

    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        """ 94.二叉树的中序遍历 """
        res = []

        def backtracking(node):
            # 终止条件
            if not node:
                return
            # 单层递归逻辑
            backtracking(node.left)
            res.append(node.val)
            backtracking(node.right)

        backtracking(root)
        return res

    """ ------------------- 深度优先遍历 迭代法 ----------------------"""
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        """ 导读：
            前序遍历 迭代
            递归调用时会将 【局部变量、函数参数、返回地址】 压入调用栈中；递归返回时会 从栈顶弹出递归参数 ---- 这是递归能返回上一层的原因
            因此可以使用 栈 实现二叉树的"""
        stack = []      # 用于实现迭代法的深度优先遍历
        res = []
        if not root:
            return res
        stack.append(root)
        while stack:
            node = stack.pop()
            res.append(node.val)
            # 入栈顺序 右、左，出栈顺序才能使 左、右
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return res

    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        """ 导读：
            中序遍历 迭代
            无法通过 前序遍历_迭代 修改而来，因为中序遍历的处理、访问顺序不同。你看，前序遍历的 中左右，处理、访问顺序是一致的
            所以 中序遍历_迭代 需要[指针]来辅助[访问] """
        stack = []
        res = []
        cur = root      # 辅助访问的指针
        while cur or stack:
            if cur:
                stack.append(cur)           # 访问
                cur = cur.left
            else:
                cur = stack.pop()
                res.append(cur.val)         # 处理
                cur = cur.right
        return res

    def postorderTraversal(self, root):
        """ 导读：
            后序遍历 迭代法
            后序遍历_迭代 可通过 前序遍历_迭代 简单修改而来：前序 中左右 -> 中右左 -> 后序 左右中"""
        stack = []
        res = []
        if not root:
            return res
        stack.append(root)
        while stack:
            node = stack.pop()
            res.append(node.val)
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
        return res[::-1]

    """ ------------------- 广度优先遍历 迭代法 ----------------------"""
    def levelOrder(self, root:Optional[TreeNode]):
        """ 导读：
            层序遍历
                广度优先遍历，使用【队列】辅助实现
                深度优先遍历，使用【栈】辅助实现 """
        queue = []
        res = []
        if not root:
            return res
        queue.append(root)
        while queue:
            length = len(queue)
            tmp = []
            for _ in range(length):
                node = queue.pop(0)
                tmp.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            if tmp:
                res.append(tmp)
        return res

    def levelOrderBottom(self, root: Optional[TreeNode]) -> List[List[int]]:
        """ 107.二叉树的层序遍历II """
        queue = []
        res = []
        if not root:
            return res
        queue.append(root)
        while queue:
            length = len(queue)
            tmp = []
            for _ in range(length):
                node = queue.pop(0)
                tmp.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            if tmp:
                res.append(tmp)
        return res[::-1]

    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        """ 199.二叉树的右视图 """
        queue = []
        res = []
        if not root:
            return res
        queue.append(root)
        while queue:
            length = len(queue)
            for _ in range(length):
                node = queue.pop(0)
                tmp = node.val
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(tmp)
        return res

    def levelOrder(self, root: 'Node') -> List[List[int]]:
        """ 429.N叉树的层序遍历 """
        queue = [root]
        res = []
        while queue:
            length = len(queue)
            tmp = []
            for _ in range(length):
                node = queue.pop(0)
                if not node:
                    continue
                tmp.append(node.val)
                queue.extend(node.children)
            if tmp:
                res.append(tmp)
        return res

    def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
        """ 116.填充每个节点的下一个右侧节点指针 """
        # 反倒不如之前提交的简洁了
        queue = []
        if not root:
            return root
        queue.append(root)
        while queue:
            length = len(queue)
            tmp = []
            for _ in range(length):
                node = queue.pop(0)
                tmp.append(node)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            for i in range(length - 1):
                tmp[i].next = tmp[i + 1]
        return root

    def connectII(self, root: 'Node') -> 'Node':
        """ 117.填充每个节点的下一个右侧节点指针II
            与 116. 完全相同"""
        # 之前提交的思路
        queue = [root]
        while queue:
            length = len(queue)
            tmp = []
            for _ in range(length):
                node = queue.pop(0)
                if not node:
                    continue
                if tmp:
                    tmp[-1].next = node
                tmp.append(node)
                queue.extend([node.left, node.right])
        return root

    def maxDepth(self, root: Optional[TreeNode]) -> int:
        """ 104.二叉树的最大深度 """
        if not root:
            return 0
        queue = [root]
        layerNum = 0
        while queue:
            length = len(queue)
            for _ in range(length):
                node = queue.pop(0)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            layerNum += 1       # 因为空节点不会进入，所以直接+=1即可
        return layerNum

    def minDepth(self, root: Optional[TreeNode]) -> int:
        """ 111.二叉树的最小深度
            层序遍历，发现叶子结点就结束 """
        minLayerNum = 0
        if not root:
            return minLayerNum
        queue = [root]
        while queue:
            length = len(queue)
            for _ in range(length):
                node = queue.pop(0)
                if not (node.left or node.right):
                    return minLayerNum + 1
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            minLayerNum += 1

    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """ 226.翻转二叉树
            尝试 层序遍历，过了 """
        # 自己写的 层序遍历
        # if not root:
        #     return root
        # queue = [root]
        # while queue:
        #     length = len(queue)
        #     for _ in range(length):
        #         node = queue.pop(0)
        #         node.left, node.right = node.right, node.left
        #         if node.left:
        #             queue.append(node.left)
        #         if node.right:
        #             queue.append(node.right)
        # return root

        # 《代码随想录》前序递归
        # def backtracking(node):
        #     # 终止条件
        #     if not node:
        #         return
        #     # 单层递归
        #     node.left, node.right = node.right, node.left
        #     backtracking(node.left)
        #     backtracking(node.right)
        #
        # backtracking(root)
        # return root

        # 《代码随想录》前序迭代
        # 迭代 实现 深度优先遍历——用【栈】来辅助
        if not root:
            return root
        stack = [root]
        while stack:
            node = stack.pop()
            node.left, node.right = node.right, node.left
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return root

    def preorder(self, root: 'Node') -> List[int]:
        """ 589.N叉树的前序遍历 """
        # 前序迭代
        # if not root:
        #     return []
        # stack = [root]
        # res = []
        # while stack:
        #     node = stack.pop()
        #     res.append(node.val)
        #     for child in node.children[::-1]:       # 从右往左添加，出栈访问才能从左往右，我想是的
        #         if not child:
        #             continue
        #         stack.append(child)
        # return res

        # 前序递归
        res = []

        def backtracking(node):
            # 终止条件
            if not node:
                return
            # 单层递归
            res.append(node.val)
            for child in node.children:       # 保持 '中左右' 的递归顺序
                backtracking(child)

        backtracking(root)
        return res

    def postorder(self, root: 'Node') -> List[int]:
        """ 590.N叉树的后序遍历
            后序迭代 = 基于前序迭代，中左右 -> 改变入栈顺序 中右左 -> 反转结果 左右中 """
        # 后序迭代
        # if not root:
        #     return []
        # stack = [root]
        # res = []
        # while stack:
        #     node = stack.pop()
        #     res.append(node.val)
        #     for child in node.children:     # 后续迭代，入栈访问顺序 从左到右，出栈(处理)顺序才能 从右到左
        #         if not child:
        #             continue
        #         stack.append(child)
        # return res[::-1]

        # 后序递归
        res = []

        def backtracking(node):
            # 终止条件
            if not node:
                return
            # 单层递归
            for child in node.children:
                if not child:
                    continue
                backtracking(child)

            res.append(node.val)

        backtracking(root)
        return res

    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        """ 101.对称二叉树
            感觉比较难的。使用后续遍历，要比较的是左右2个子树，而不只是左右2个节点 """
        # 《代码随想录》递归
        # def backtracking(left, right):      # 参数：左、右2个子树 返回值：bool
        #     # 终止条件
        #     # 处理 空 的情况
        #     if not left and right:          # 左空
        #         return False
        #     elif left and not right:        # 右空
        #         return False
        #     elif not (left or right):       # 左、右空
        #         return True
        #     # 上面处理了所有 空 的情况，下面就都 不空 了
        #     elif left.val != right.val:     # 值不等
        #         return False
        #     else:                           # 值相等
        #         # 结合看两个递归的 第一个参数——左子树的遍历顺序 左右中
        #         #               第二个参数——右子树的遍历顺序 右左中
        #         outside = backtracking(left.left, right.right)      # 比较外侧节点
        #         inside = backtracking(left.right, right.left)       # 比较内侧节点
        #         return outside and inside
        #
        # return backtracking(root.left, root.right)

        # 《代码随想录》迭代法
        # 这可不是层序遍历！
        if not root:
            return True
        queue = [root.left, root.right]
        while queue:
            left = queue.pop(0)
            right = queue.pop(0)
            # 判断条件与递归时相同
            # 处理 空 的情况
            if not (left or right):         # 均 空，不是直接返回True，而是要继续判断
                continue
            elif not left or not right:     # 一个空，那不对称
                return False
            elif left.val != right.val:     # 没有空，但值不等
                return False
            else:                           # 没有空，值等
                queue.extend([left.left, right.right, left.right, right.left])
        return True

    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        """ 100.相同的树
            解题思路同上 101.对称二叉树 """
        # 递归法
        def backtracking(first, second):
            # 终止条件
            if not (first or second):           # 均空
                return True
            elif not (first and second):        # 一个空
                return False
            elif first.val != second.val:       # 都不空，值不等
                return False
            else:                               # 都不空，值等，继续往后
                left_half = backtracking(first.left, second.left)
                right_half = backtracking(first.right, second.right)
                return left_half and right_half

        return backtracking(p, q)

        # 迭代法
        # if not (p or q):
        #     return True
        # elif not (p and q):
        #     return False
        # queue = [p, q]
        # while queue:
        #     first = queue.pop(0)
        #     second = queue.pop(0)
        #     # 判断条件同递归时
        #     if not (first or second):         # 均空
        #         continue
        #     elif not (first and second):      # 一个空
        #         return False
        #     elif first.val != second.val:     # 都不空，但值不等
        #         return False
        #     else:                             # 都不空，值等
        #         queue.extend([first.left, second.left, first.right, second.right])
        # return True

    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        """ 572.另一棵树的子树
            参考 100.相同的树
            答案：https://leetcode.cn/problems/subtree-of-another-tree/solutions/235634/dui-cheng-mei-pan-duan-zi-shu-vs-pan-duan-xiang-de/"""
        def backtracking(first, second):
            """ 100.相同的树 """
            # 终止条件
            if not (first or second):       # 均空
                return True
            elif not (first and second):    # 一个空
                return False
            elif first.val != second.val:   # 都不空，值不等
                return False
            else:                           # 都不空，值等
                left_of_both = backtracking(first.left, second.left)
                right_of_both = backtracking(first.right, second.right)
                return left_of_both and right_of_both

        if not (root or subRoot):
            return True
        elif not (root and subRoot):
            return False
        return backtracking(root, subRoot) or self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)

    def maxDepth(self, root: Optional[TreeNode]) -> int:
        """ 104.二叉树的最大深度 """
        # 《代码随想录》层序遍历
        # 比自己写要简洁很多
        # if not root:
        #     return 0
        # queue = [root]
        # maxDepth = 0
        # while queue:
        #     length = len(queue)
        #     maxDepth += 1
        #     for _ in range(length):
        #         node = queue.pop(0)
        #         if node.left:
        #             queue.append(node.left)
        #         if node.right:
        #             queue.append(node.right)
        # return maxDepth

        # 《代码随想录》后序递归遍历
        # 前序递归 求 深度
        # 后序递归 求 高度
        # 根节点的高度 就是 二叉树的深度，所以用后序遍历求根节点的高度
        # def backtracking(node):
        #     """ 后序递归 求 节点的高度 """
        #     # 终止条件
        #     if not node:
        #         return 0
        #     # 单层递归
        #     left_height = backtracking(node.left)
        #     right_height = backtracking(node.right)
        #     return max(left_height, right_height) + 1       # 返回以当前节点node为根节点的树的高度
        #
        # return backtracking(root)

        # 《代码随想录》前序递归遍历，充分体现回溯
        # 自己可能不太好想出来
        res = 0

        def backtracking(node, depth):
            nonlocal res
            res = max(res, depth)
            # 终止条件
            if not (node.left or node.right):
                return      # 递归过程中已经记录了最大深度res，所以不用返回
            # 单层递归逻辑
            if node.left:
                depth += 1
                backtracking(node.left, depth)
                depth -= 1
            if node.right:
                depth += 1
                backtracking(node.right, depth)
                depth -= 1
            return
        if not root:
            return 0
        backtracking(root, 1)
        return res

    def maxDepth(self, root: 'Node') -> int:
        """ 559.N叉树的最大深度
            首推 后序递归遍历 """
        # 104.二叉树的最大深度 后序递归遍历，感觉这个是最简单的
        # def backtracking(node):
        #     """ 后序递归遍历求当前节点的高度 """
        #     # 终止条件
        #     if not node:
        #         return 0
        #     # 单层递归
        #     every_height = []
        #     for child in node.children:
        #         every_height.append(backtracking(child))
        #     return max(every_height) + 1 if every_height else 1
        #
        # if not root:
        #     return 0
        # return backtracking(root)

        # 层序遍历
        # 权当加深印象吧
        if not root:
            return 0
        queue = [root]
        maxDepth = 0
        while queue:
            length = len(queue)
            maxDepth += 1
            for _ in range(length):
                node = queue.pop(0)
                for child in node.children:
                    if child:
                        queue.append(child)
        return maxDepth

    def minDepth(self, root: Optional[TreeNode]) -> int:
        """ 111.二叉树的最小深度
            首推《代码随想录》递归后序遍历 """
        # 层序遍历
        # if not root:
        #     return 0
        # queue = [root]
        # minDepth = 0
        # while queue:
        #     length = len(queue)
        #     minDepth += 1
        #     for _ in range(length):
        #         node = queue.pop(0)
        #         if not (node.left or node.right):
        #             return minDepth
        #         if node.left:
        #             queue.append(node.left)
        #         if node.right:
        #             queue.append(node.right)
        # return minDepth

        # 递归后序遍历，遇到叶子节点返回
        # 《代码随想录》实现略有差别，会更简单
        # def backtracking(node):
        #     # 终止条件
        #     if not node:
        #         return 0
        #     if not (node.left or node.right):
        #         return 1
        #     # 单层递归
        #     left_height = backtracking(node.left)
        #     right_height = backtracking(node.right)
        #     if node.left and node.right:
        #         return min(left_height, right_height) + 1
        #     elif not (node.left and node.right):
        #         return left_height + 1 if node.left else right_height + 1
        #
        # return backtracking(root)

        # 《代码随想录》前序递归遍历
        # 不太推荐 前序递归，不太好理解，从公式角度来看也不太好记
        minDepth = float('inf')

        def backtrakcing(node, depth):
            """ 前序递归遍历 求 最小深度 """
            # 终止条件
            if not node:
                return
            # 单层递归
            if not (node.left or node.right):       # 当前节点是叶子节点
                nonlocal minDepth
                minDepth = min(minDepth, depth)
            if node.left:
                backtrakcing(node.left, depth + 1)
            if node.right:
                backtrakcing(node.right, depth + 1)

        if not root:
            return 0
        backtrakcing(root, 1)
        return minDepth

    def countNodes(self, root: Optional[TreeNode]) -> int:
        """ 222.完全二叉树的节点个数 """
        # 层序遍历
        # if not root:
        #     return 0
        # queue = [root]
        # counts = 0
        # while queue:
        #     length = len(queue)
        #     for _ in range(length):
        #         node = queue.pop(0)
        #         counts += 1
        #         if node.left:
        #             queue.append(node.left)
        #         if node.right:
        #             queue.append(node.right)
        # return counts

        # 前序递归遍历
        # counts = 0
        #
        # def backtracking(node):
        #     # 终止条件
        #     if not node:
        #         return
        #     # 单层递归
        #     nonlocal counts
        #     counts += 1
        #     backtracking(node.left)
        #     backtracking(node.right)
        #
        # backtracking(root)
        # return counts

        # 后序递归遍历
        # 参考 104.二叉树的最大深度 的后序递归
        def backtracking(node):
            """ 返回 以当前节点node为根节点的树 的节点数量 """
            # 终止条件
            if not node:
                return 0
            # 单层递归
            left_counts = backtracking(node.left)
            right_counts = backtracking(node.right)
            return left_counts + right_counts + 1

        return backtracking(root)

    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        """ 110.平衡二叉树
            与 104.二叉树的最大深度 不完全相同：
                104.二叉树的最大深度：仅仅求node为根节点 树的最大深度
                110.平衡二叉树：若递归过程中发现不是平衡二叉树了，返回树的高度也没意义了，用-1表示已经不是平衡二叉树了 """
        def backtracking(node):
            """ 基于 104.二叉树的最大深度，递归过程中若发现已经不是平衡二叉树了 则返回-1 """
            # 终止条件
            if not node:
                return 0
            # 单层递归
            left_h = backtracking(node.left)
            right_h = backtracking(node.right)
            if -1 in [left_h, right_h]:
                return -1
            if abs(left_h - right_h) > 1:       # 此时已经发现不是平衡二叉树了，返回树的高度也没意义了，返回-1代表不是平衡二叉树就行了
                return -1
            return max(left_h, right_h) + 1

        return True if backtracking(root) != -1 else False

    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
        """ 257.二叉树的所有路径
            本题要找所有路径，前序遍历最合适，方便父节点指向子节点
            二叉树 第一次使用回溯 """
        # 前序递归
        # 代码可以再精简一些，下次吧
        res = []
        path = []

        def backtracking(node):
            # 终止条件
            # 不用考虑空节点，因为后面控制空节点不进入递归
            if not (node.left or node.right):
                path.append(node.val)
                res.append("->".join([str(n) for n in path]))
                return
            # 单层递归
            path.append(node.val)
            if node.left:
                backtracking(node.left)
                path.pop()
            if node.right:
                backtracking(node.right)
                path.pop()

        backtracking(root)
        return res

    def sumOfLeftLeaves(self, root: Optional[TreeNode]) -> int:
        """ 404.左叶子之和
            就不能用层序遍历，因为要通过 当前节点的父节点 才能判断 当前节点 是不是左叶子 """
        # 《代码随想录》后序递归遍历
        # 因为要通过递归函数的返回值累加求取左叶子之和(似乎用到递归函数的返回值 就是 后序递归？)
        # def backtracking(node):
        #     """ 以当前节点node为根节点的树 的 左叶子之和 """
        #     # 终止条件
        #     # 只有当前节点node是父节点，才有可能判断出左叶子
        #     if not node:
        #         return 0
        #     elif not (node.left or node.right):
        #         return 0
        #     # 单层递归
        #     left_res = backtracking(node.left)      # 以node.left为根节点的树的左叶子之和；node.left可能是左叶子，下面判断
        #     if node.left and not (node.left.left or node.left.right):
        #         left_res = node.left.val
        #     right_res = backtracking(node.right)    # 以node.right为根节点的树的左叶子之和；node.right不可能是左叶子
        #     return left_res + right_res
        #
        # return backtracking(root)

        # 《代码随想录》迭代，前中后均可，判断条件相同
        # 前序迭代
        stack = [root]
        res = 0
        while stack:
            node = stack.pop()
            # 判断条件相同
            if not node:
                continue
            elif not (node.left or node.right):
                continue
            if node.left and not (node.left.left or node.left.right):
                res += node.left.val
            stack.extend([node.right, node.left])
        return res

    def findBottomLeftValue(self, root: Optional[TreeNode]) -> int:
        """ 513.找树左下角的值 """
        # 迭代最简单
        # queue = [root]
        # while queue:
        #     length = len(queue)
        #     for i in range(length):
        #         node = queue.pop(0)
        #         if i == 0:
        #             res = node.val
        #         if node.left:
        #             queue.append(node.left)
        #         if node.right:
        #             queue.append(node.right)
        # return res

        # 《代码随想录》前序递归，当然感觉不是太好理解，又当然下面自己写的，竟然过了
        maxDepth = float('-inf')
        res = 0

        def backtracking(node, depth):
            # 终止条件
            if not node:
                return 0
            # 单层递归
            nonlocal maxDepth
            nonlocal res
            if depth > maxDepth:
                maxDepth = depth
                res = node.val
            backtracking(node.left, depth + 1)
            backtracking(node.right, depth + 1)
        backtracking(root, 1)
        return res

    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        """ 112.路径总和
            参考 257.二叉树的所有路径--二叉树第1次使用回溯
            递归函数什么时候需要返回值：
                1.递归遍历整棵树 且 不需要处理返回值，递归函数就不要返回值
                2.递归遍历整棵树 且 需要处理返回值，递归函数就需要返回值
                3.搜索一条符合条件的路径，则需要返回值，因为找到需要及时返回 """
        # 《代码随想录》找一条符合条件的路径，递归函数需要返回值，找到立刻返回
        def backtracking(node, count):
            """" 从以node为根节点的树上找，count初始为targetSum """
            # 终止条件
            if not (node.left or node.right) and count == 0:
                return True
            elif not (node.left or node.right):
                return False
            # 单层递归
            if node.left:
                if backtracking(node.left, count - node.left.val):      # 隐藏的回溯
                    return True     # 找到符合条件的路径，及时返回
            if node.right:
                if backtracking(node.right, count - node.right.val):    # 隐藏的回溯
                    return True     # 找到符合条件的路径，及时返回
            return False
        if not root:
            return False
        return backtracking(root, targetSum)


        # 前序递归遍历，求所有路径。自己写的虽然过了，感觉比较麻烦
        # res = False
        # path = []
        #
        # def backtracking(node):
        #     # 终止条件
        #     if not (node.left or node.right):
        #         path.append(node.val)
        #         if sum(path) == targetSum:
        #             nonlocal res
        #             res = True
        #         return
        #     # 单层递归
        #     path.append(node.val)
        #     if node.left:
        #         backtracking(node.left)
        #         path.pop()
        #     if node.right:
        #         backtracking(node.right)
        #         path.pop()
        #
        # if not root:
        #     return False
        # backtracking(root)
        # return res

    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        """ 106.从中序与后序遍历序列构造二叉树 """
        if not postorder:
            return None
        rootVal = postorder[-1]
        root = TreeNode(rootVal)
        root_inorder_ind = inorder.index(rootVal)
        root.left = self.buildTree(inorder[:root_inorder_ind][:], postorder[:root_inorder_ind][:])
        root.right = self.buildTree(inorder[root_inorder_ind + 1:], postorder[root_inorder_ind:-1][:])
        return root

    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        """ 105.从前序与中序遍历序列构造二叉树 """
        if not inorder:
            return None
        rootVal = preorder[0]
        root = TreeNode(rootVal)
        root_inorder_ind = inorder.index(rootVal)
        root.left = self.buildTree(preorder[1:root_inorder_ind + 1][:], inorder[:root_inorder_ind])
        root.right = self.buildTree(preorder[root_inorder_ind + 1:][:], inorder[root_inorder_ind + 1:][:])
        return root

    def constructMaximumBinaryTree(self, nums: List[int]) -> Optional[TreeNode]:
        """ 654.最大二叉树 """
        if not nums:
            return None
        rootVal = max(nums)
        root = TreeNode(rootVal)
        rootValInd = nums.index(rootVal)
        root.left = self.constructMaximumBinaryTree(nums[:rootValInd])
        root.right = self.constructMaximumBinaryTree(nums[rootValInd + 1:])
        return root

    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
        """ 617.合并二叉树 """
        # 《代码随想录》递归，前中后序递归都可以
        # def backtracking(node1, node2):
        #     if not node1:
        #         return node2
        #     elif not node2:
        #         return node1
        #     # 上面判断是否有空，执行到这里就是都不空
        #     # 体现前序递归遍历
        #     node1.val += node2.val
        #     node1.left = backtracking(node1.left, node2.left)
        #     node1.right = backtracking(node1.right, node2.right)
        #     return node1
        #
        # return backtracking(root1, root2)

        # 《代码随想录》迭代法，同时遍历2棵树
        if not root1:
            return root2
        elif not root2:
            return root1
        queue = [[root1, root2]]            # 不空的节点才进队列
        while queue:
            node1, node2 = queue.pop(0)     # 队列取出的肯定不空
            node1.val += node2.val
            # 2棵树的左、右节点均不空，同时入队列
            if node1.left and node2.left:
                queue.append([node1.left, node2.left])
            if node1.right and node2.right:
                queue.append([node1.right, node2.right])
            # 若node1空、node2不空，node2直接给node1；node1不空、node2空的就不处理了，最后返回node1嘛
            if not node1.left and node2.left:
                node1.left = node2.left
            if not node1.right and node2.right:
                node1.right = node2.right
        return root1

    def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        """ 700.二叉搜索树中的搜索 """
        # 《代码随想录》迭代法
        while root:
            if root.val == val:
                return root
            elif root.val > val:
                root = root.left
            else:
                root = root.right
        return None     # 返回root也行，毕竟到这里就是空了


        # 《代码随想录》递归 使用BST的特性
        # def backtracking(node):
        #     # 终止条件
        #     if not node or node.val == val:
        #         return node
        #     # 单层递归
        #     if node.val > val:      # 则去左子树找
        #         return backtracking(node.left)
        #     else:                   # 则去右子树找
        #         return backtracking(node.right)
        #
        # return backtracking(root)

        # 递归 自己写的，没有使用BST的特性
        # def backtracking(node):
        #     # 终止条件
        #     if not node:
        #         return None
        #     elif node.val == val:
        #         return node
        #     # 单层递归
        #     left = backtracking(node.left)
        #     right = backtracking(node.right)
        #     return left if left else right
        #
        # return backtracking(root)

    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        """ 98.验证二叉搜索树
            BST中序遍历严格递增 """
        # 迭代法
        # stack = []
        # cur = root
        # res = []
        # while cur or stack:
        #     if cur:
        #         stack.append(cur)
        #         cur = cur.left
        #     else:
        #         node = stack.pop()
        #         if res and res[-1] >= node.val:
        #             return False
        #         res.append(node.val)
        #         cur = node.right
        # return True

        # 中序递归遍历
        res = []

        def backtracking(node):
            # 终止条件
            if not node:
                return
            # 单层递归
            backtracking(node.left)
            res.append(node.val)
            backtracking(node.right)

        backtracking(root)
        for i in range(1, len(res)):
            if res[i - 1] >= res[i]:
                return False
        return True

    def getMinimumDifference(self, root: Optional[TreeNode]) -> int:
        """ 530.二叉搜索树的最小绝对差 """
        # 《代码随想录》BST中序递归
        # res = []
        # minSubVal = float('inf')
        #
        # def backtracking(node):
        #     # 终止条件
        #     if not node:
        #         return
        #     # 单层递归
        #     backtracking(node.left)
        #     if res:
        #         nonlocal minSubVal
        #         minSubVal = min(minSubVal, node.val - res[-1])
        #     res.append(node.val)
        #     backtracking(node.right)
        #
        # backtracking(root)
        # return minSubVal

        # 《代码随想录》BST中序迭代法
        # 尝试用下临时变量，而不是记录所有值
        stack = []
        cur = root
        pre = None          # 只记录前一个节点，而不是记录所有已访问的节点
        minSubVal = float('inf')
        while cur or stack:
            if cur:
                stack.append(cur)
                cur = cur.left
            else:
                node = stack.pop()
                if pre:
                    minSubVal = min(minSubVal, node.val - pre.val)
                pre = node
                cur = node.right
        return minSubVal

    def findMode(self, root: Optional[TreeNode]) -> List[int]:
        """ 501.二叉搜索数中的众数 """
        # from collections import defaultdict
        #
        # # 《代码随想录》递归
        # val2counts = defaultdict(int)
        # def backtracking(node):
        #     # 终止条件
        #     if not node:
        #         return
        #     val2counts[node.val] += 1
        #     # 单层递归
        #     backtracking(node.left)
        #     backtracking(node.right)
        #
        # backtracking(root)
        # maxCount = max(val2counts.keys())         # 看周总结时看到这里代码，很可能是 .values()
        # res = [k for k, v in val2counts.items() if v == maxCount]
        # return res

        # 《代码随想录》递归 利用BST中序遍历升序
        # res = []
        # pre = None      # 指示前一个访问的节点
        # # cur = None      # 指示当前访问节点
        # count = 0       # 当前元素出现次数
        # maxCount = 0
        #
        # def backtracking(node):
        #     # 终止条件
        #     if not node:
        #         return
        #     # 单层递归
        #     backtracking(node.left)
        #     nonlocal pre, count, maxCount
        #     if not pre:             # pre空，节点首次出现
        #         count = 1
        #     elif pre.val == node.val:
        #         count += 1
        #     elif pre.val != node.val:
        #         count = 1
        #     pre = node
        #     # 收集节点
        #     if count == maxCount:
        #         res.append(node.val)
        #     elif count > maxCount:
        #         maxCount = count
        #         res.clear()
        #         res.append(node.val)
        #     backtracking(node.right)
        #
        # backtracking(root)
        # return res

        # 《代码随想录》迭代法-->不要想当然层序遍历
        # 中序遍历 迭代法；中间节点的处理逻辑一模一样
        stack = []
        cur = root
        pre = None
        count = 0
        maxCount = 0
        res = []
        while stack or cur:
            if cur:
                stack.append(cur)
                cur = cur.left
            else:
                cur = stack.pop()
                if not pre:             # pre空，意味着节点首次出现
                    count = 1
                elif pre.val == cur.val:
                    count += 1
                elif pre.val != cur.val:
                    count = 1
                pre = cur
                # 收集节点
                if count == maxCount:
                    res.append(cur.val)
                elif count > maxCount:
                    maxCount = count
                    res.clear()
                    res.append(cur.val)

                cur = cur.right         # 继续中序遍历
        return res

    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        """ 236.二叉树的最近公共祖先
            找公共祖先，能自底向上最好，回溯天然就是自底向上，回溯后续 左右中 更加符合用左右节点返回值处理中间节点的逻辑 """
        # 因为返回值刚好需要TreeNode，直接利用题目代码作回溯递归函数
        # 终止条件
        # if not root or root in [p, q]:      # 子树中遇到p或q就返回，遇到空就是终止条件也返回
        #     return root
        # # 单层递归
        # # 因为要递归整棵树(原因看题解)，所以用left、right接住递归结果
        # left = self.lowestCommonAncestor(root.left, p, q)
        # right = self.lowestCommonAncestor(root.right, p, q)
        # # 中间节点处理逻辑：无论是左子树还是右子树找到的p或q，都向上返回；都没找到则返回空
        # if left and right:      # 左右返回都不空，就是左右都找到了，当前节点root就是最近公共祖先
        #     return root
        # if left and not right:
        #     return left
        # elif not left and right:
        #     return right
        # elif not (left or right):
        #     return None

        # 再来一遍
        # 找公共祖先，想到自底向上，回溯天然是自底向上，回溯后序遍历左右中 则更符合不过了
        # 返回值也是TreeNode，因此直接使用题目函数作递归函数
        # 终止条件
        if not root or root in [p, q]:      # 遇到空节点返回；找到p/q返回
            return root
        # 单层递归 后序遍历，左右中
        # 因为遍历整棵树，所以用left、right接住递归结果用以继续处理
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        # 处理中节点
        if left and right:      # 左右子树都找到了p/q
            return root
        # 将不空的结果向上返回
        if left or not right:
            return left
        elif not left or right:
            return right
        elif not (left or right):
            return None

    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        """ 235.二叉搜索树的最近公共祖先
            一定利用BST特性：左子树节点值<根节点值<右子树节点值
            二叉树递归返回值：
                1.搜索一条路径，找到直接返回
                2.搜索整棵树，接受左右子树处理结果，然后处理返回 """
        # 《代码随想录》递归，前中后序递归均可，因为不需要处理中间节点，找到就行了
        # def backtracking(node):
        #     # 终止条件
        #     if not node:
        #         return node
        #     # 单层递归
        #     if node.val > max(p.val, q.val):        # 则去 左子树 找
        #         # 这里直接把递归返回值 left 返回，因为只要找到符合的值就行，下同
        #         left = backtracking(node.left)
        #         if left:                            # 找到就行了
        #             return left
        #     if node.val < min(p.val, q.val):        # 则去 右子树 找
        #         right = backtracking(node.right)
        #         if right:                           # 找到就行了
        #             return right
        #     return node                             # 那就是 node.val 介于二者之间，直接返回
        #
        # return backtracking(root)

        # 《代码随想录》迭代法，逻辑类似递归
        while root:
            if root.val > max(p.val, q.val):
                root = root.left
            elif root.val < min(p.val, q.val):
                root = root.right
            else:                           # root.val介于二者之间，找到了，直接返回
                return root
        return None                         # 没找到

    def insertIntoBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        """ 701.二叉搜索树中的插入操作 """
        if not root:
            return TreeNode(val)
        cur = root
        # pre = None
        while cur:
            pre = cur
            if cur.val > val:
                cur = cur.left
            elif cur.val < val:
                cur = cur.right
            if not cur:
                if pre.val < val:
                    pre.right = TreeNode(val)
                else:
                    pre.left = TreeNode(val)
        return root

if __name__ == "__main__":
    pass

"""
递归三要素：
    1.递归函数参数和返回值
    2.终止条件
    3.单层递归逻辑
"""