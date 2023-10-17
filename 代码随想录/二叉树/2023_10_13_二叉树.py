from typing import Optional, List


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Node:
    def __init__(self, val=0, children=None):
        self.val = val
        self.children = children


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

        # 重写递归
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
        # stack = []
        # res = []
        # if not root:
        #     return res
        # stack.append(root)
        # while stack:
        #     node = stack.pop()
        #     res.append(node.val)
        #     if node.right:
        #         stack.append(node.right)
        #     if node.left:
        #         stack.append(node.left)
        # return res

        # 重写迭代
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

        # 重写递归
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
        # res = []
        # stack = []
        # node = root
        # while node or stack:
        #     if node:
        #         stack.append(node)
        #         node = node.left
        #     else:
        #         node = stack.pop()
        #         res.append(node.val)
        #         node = node.right
        # return res

        # 重写迭代
        stack = []
        res = []
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

        # 重写递归
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
        # res = []
        # stack = []
        # if not root:
        #     return res
        # stack.append(root)
        # while stack:
        #     node = stack.pop()
        #     res.append(node.val)
        #     if node.left:
        #         stack.append(node.left)
        #     if node.right:
        #         stack.append(node.right)
        # return res[::-1]

        # 重写迭代
        stack = []
        res = []
        if not root:
            return res
        stack.append(root)
        while stack:
            node = stack.pop()
            res.append(node.val)
            if node.left:
                res.append(node.left)
            if node.right:
                res.append(node.right)
        return res[::-1]

    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        """ 102.二叉树的层序遍历
            中等 """
        # queue = [root]      # 队列
        # res = []
        # while queue:
        #     length = len(queue)
        #     tmp = []
        #     for _ in range(length):
        #         node = queue.pop(0)
        #         if not node:
        #             continue
        #         tmp.append(node.val)
        #         queue.extend([node.left, node.right])
        #     if tmp:
        #         res.append(tmp)
        # return res

        # 再写一遍 层序遍历
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

    def levelOrderBottom(self, root: Optional[TreeNode]) -> List[List[int]]:
        """ 107.二叉树的层序遍历II
            中等 """
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
        return res[::-1]

    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        """ 199.二叉树的右视图
            中等
            思路：层序遍历 """
        # queue = [root]
        # res = []
        # while queue:
        #     length = len(queue)
        #     tmp = []
        #     for _ in range(length):
        #         node = queue.pop(0)
        #         if not node:
        #             continue
        #         tmp.append(node.val)
        #         queue.extend([node.left, node.right])
        #     if tmp:
        #         res.append(tmp)
        # return [ls[-1] for ls in res]

        # 答案思路，只收集单层最后一个元素
        # 这个思路queue只能进非空node
        queue = []
        res = []
        if not root:
            return res
        queue.append(root)
        while queue:
            length = len(queue)
            for i in range(length):
                node = queue.pop(0)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
                if i == length - 1:
                    res.append(node.val)
        return res

    def averageOfLevels(self, root: Optional[TreeNode]) -> List[float]:
        """ 637.二叉树的层平均值
            简单 """
        queue = [root]      # 注意细节，我这里node空不空都能进queue，有的层序遍历只能进非空node
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
        return [sum(ls)/len(ls) for ls in res]

    def levelOrder(self, root: 'Node') -> List[List[int]]:
        """ 429.N叉树的层序遍历
            中等 """
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


if __name__ == "__main__":
    sl = Solution()

    """
    递归三要素：
        1.递归函数的参数和返回值
        2.终止条件
        3.单层递归逻辑
    """