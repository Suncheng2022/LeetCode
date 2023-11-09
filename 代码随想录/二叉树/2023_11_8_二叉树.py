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

    """--->再看8.对称二叉树"""


if __name__ == "__main__":
    pass

"""
递归三要素：
    1.递归函数参数和返回值
    2.终止条件
    3.单层递归逻辑
"""