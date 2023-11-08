from typing import Optional, List


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

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



if __name__ == "__main__":


"""
递归三要素：
    1.递归函数参数和返回值
    2.终止条件
    3.单层递归逻辑
"""