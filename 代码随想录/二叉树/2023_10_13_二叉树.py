from typing import Optional, List


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        """ 144.二叉树的前序遍历
            简单 """
        # 递归
        # res = []
        #
        # def backtracking(node):
        #     if not node:
        #         return
        #     res.append(node.val)
        #     backtracking(node.left)
        #     backtracking(node.right)
        #
        # backtracking(root)
        # return res

        # 迭代
        stack = []
        res = []
        if not root:
            return res
        stack.append(root)
        while stack:
            node = stack.pop()
            res.append(node.val)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return res

    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        """ 94.二叉树的中序遍历
            简单 """
        # 递归
        # res = []
        #
        # def backtracking(node):
        #     if not node:
        #         return
        #     backtracking(node.left)
        #     res.append(node.val)
        #     backtracking(node.right)
        #
        # backtracking(root)
        # return res

        # 迭代
        res = []
        stack = []
        node = root
        while node or stack:
            if node:
                stack.append(node)
                node = node.left
            else:
                node = stack.pop()
                res.append(node.val)
                node = node.right
        return res

    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        """ 145.二叉树的后序遍历
            简单 """
        # 递归
        # res = []
        #
        # def backtracking(node):
        #     if not node:
        #         return
        #     backtracking(node.left)
        #     backtracking(node.right)
        #     res.append(node.val)
        #
        # backtracking(root)
        # return res

        # 迭代 由 先序遍历的 中左右 -> 修改为 中右左 -> 反转遍历结果，得到后序遍历的 左右中
        res = []
        stack = []
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


if __name__ == "__main__":
    sl = Solution()

    root = [1, None, 2, 3]
    print(sl.inorderTraversal(root))

    """
    递归三要素：
        1.递归函数的参数和返回值
        2.终止条件
        3.单层递归逻辑
    """