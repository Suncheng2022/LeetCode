from typing import Optional, List


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        """ 144.二叉树的前序遍历 """
        # 递归前序遍历
        # res = []
        #
        # def backtracking(node):
        #     # 终止条件
        #     if not node:
        #         return
        #     # 单层递归
        #     res.append(node.val)
        #     backtracking(node.left)
        #     backtracking(node.right)
        #
        # backtracking(root)
        # return res

        # 迭代前序遍历
        stack = [root]
        res = []
        while stack:
            node = stack.pop()
            if not node:
                continue
            res.append(node.val)
            stack.extend([node.right, node.left])       # 注意入栈顺序，右、左入栈，访问才能左、右
        return res

    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        """ 145.二叉树的后序遍历 """
        # 递归后序遍历
        # res = []
        #
        # def backtracking(node):
        #     # 终止条件
        #     if not node:
        #         return
        #     # 单层递归
        #     backtracking(node.left)
        #     backtracking(node.right)
        #     res.append(node.val)
        #
        # backtracking(root)
        # return res

        # 迭代后序遍历 左右中
        # 前序 中左右 --> 中右左 --> 左右中
        stack = [root]
        res = []
        while stack:
            node = stack.pop()
            if not node:
                continue
            res.append(node.val)
            stack.extend([node.left, node.right])     # 注意入栈顺序，先左、右，出栈访问才能右、左
        return res[::-1]

    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        """ 94.二叉树的中序遍历 """
        # 递归中序遍历
        # res = []
        #
        # def backtracking(node):
        #     # 终止条件
        #     if not node:
        #         return
        #     # 单层递归
        #     backtracking(node.left)
        #     res.append(node.val)
        #     backtracking(node.right)
        #
        # backtracking(root)
        # return res

        # 迭代中序遍历
        # 可不是迭代前序随便改改就能行的
        if not root:
            return []
        stack = []
        cur = root
        res = []
        while cur or stack:
            if cur:
                stack.append(cur)
                cur = cur.left
            else:
                cur = stack.pop()
                res.append(cur.val)
                cur = cur.right
        return res


if __name__ == "__main__":
    pass

"""
递归的实现：将函数参数值、局部变量、返回地址压入stack中，递归返回，即从stack顶弹出上一次递归的各项参数
"""