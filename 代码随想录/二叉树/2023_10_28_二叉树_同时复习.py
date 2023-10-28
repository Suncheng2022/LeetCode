""" 回来复习二叉树，否则边学边忘，事倍功半 """
from typing import Optional, List


class TreeNode:
    def __int__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    """
    ---------------- 前序遍历 的 递归实现 ----------------
    稍微改变顺序就可以改为 中序递归遍历、后序递归遍历。示例见下面，
    """
    def preorderTraversal(self, root):
        """ 前序遍历 递归三部曲 """
        res = []

        def backtracking(node):
            # 终止条件
            if not node:
                return
            # 单层搜索
            res.append(node.val)
            backtracking(node.left)
            backtracking(node.right)

        backtracking(root)
        return res

    def inorderTraversal(self, root):
        """ 中序遍历 递归三部曲 """
        res = []

        def backtracking(node):
            # 终止条件
            if not node:
                return
            # 单层搜索
            backtracking(node.left)
            res.append(node.val)
            backtracking(node.right)

        backtracking(root)
        return res

    def postorderTraversal(self, root):
        """ 后续遍历 递归 """
        res = []

        def backtracking(node):
            # 终止条件
            if not node:
                return
            # 单层搜索
            backtracking(node.left)
            backtracking(node.right)
            res.append(node.val)

        backtracking(root)
        return res

    """ 迭代法：
            二叉树的深度优先遍历--前、中、后 序遍历可由递归实现，递归 在系统中是通过 栈 实现，所以使用 栈 数据结构可以实现迭代法遍历 """
    def preorderTraversal_iteration(self, root):
        """ 前序遍历 迭代法--栈
                注意如栈顺序 """
        stack = []
        res = []
        if not root:
            return res
        stack.append(root)
        while stack:
            node = stack.pop()
            res.append(node.val)
            # 注意，前序遍历 孩子节点的入栈顺序——右、左入栈，这样才能 左、右出栈
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return res

    def inorderTraversal_iteration(self, root):
        """ 中序遍历 迭代法--栈 【中序遍历的迭代法与前、后序遍历不同】；要区分 访问/遍历(入栈)、处理(收集元素值) 的概念
                使用指针辅助遍历(进栈) """
        stack = []
        res = []
        cur = root      # 中序遍历的迭代法，需要指针辅助遍历
        while cur or stack:     # 二者有一个不空就可以
            if cur:
                stack.append(cur)
                cur = cur.left
            else:
                node = stack.pop()
                res.append(node.val)
                cur = node.right
        return res

    def postorderTraversal_iteration(self, root):
        """ 后续遍历 迭代法--栈 由 前序遍历 修改而来；前序遍历 中左右->中右左->反转，左右中
                            相当于，仅修改了 1.前序遍历_迭代法 的入栈顺序
                                          2.反转res的顺序 """
        stack = []
        res = []
        if not root:
            return res
        stack.append(root)
        while stack:
            node = stack.pop()
            res.append(node.val)
            # 入栈顺序与前序遍历不同，左、右入栈，则可以得到 右、左访问顺序
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
        return res[::-1]

    """ ---------------- 以下为 层序遍历 相关题目 ----------------- """
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        """ 102.二叉树的层序遍历
            层序遍历--队列 """
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
                queue.extend([node.left, node.right])
            if tmp:
                res.append(tmp)
        return res

    def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
        """ 116.填充每个节点的下一个右侧节点指针 """
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

    def minDepth(self, root: Optional[TreeNode]) -> int:
        """ 111.二叉树的最小深度 """
        queue = []
        depth = 0
        if not root:
            return depth
        queue.append(root)

        while queue:
            length = len(queue)
            for _ in range(length):
                node = queue.pop(0)
                if not (node.left or node.right):       # 层序遍历中遇到叶子结点则结束
                    return depth + 1
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            depth += 1
        return depth


if __name__ == '__main__':
    pass
"""
递归三部曲：
    1.递归函数的参数和返回值
    2.终止条件
    3.单层搜索

二叉树 遍历方式 系统总结：
    深度优先遍历：
        1.前序遍历 中左右 [递归、迭代]
        2.中序遍历 左中右 [递归、迭代]
        3.后续遍历 左右中 [递归、迭代]
    广度优先遍历：
        层序遍历[迭代 队列实现]
"""