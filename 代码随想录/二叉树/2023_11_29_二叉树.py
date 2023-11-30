from typing import Optional, List


# Definition for a binary tree node.
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

    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        """ 102.二叉树的层序遍历
            同时可做出来：
                107.二叉树的层序遍历II
                199.二叉树的右视图
                637.二叉树的层平均值
                429.N叉树的层序遍历
                515.在每个树行中找最大值
                116.填充每个节点的下一个右侧节点指针，稍稍绕一下
                117.填充每个节点的下一个右侧节点指针II
                104.二叉树的最大深度
                111.二叉树的最小深度，稍微有点难度 """
        # 《代码随想录》层序遍历 一个打十个
        if not root:
            return []
        queue = [root]
        res = []
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

    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """ 226.翻转二叉树 """
        # 感觉随便一种遍历就能做到，只要遍历的时候交换当前节点的左右子节点；《代码随想录》说，唯独中序不行，自己画一下
        # 递归先序遍历
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

        # 迭代先序遍历
        if not root:
            return root
        stack = [root]
        while stack:
            node = stack.pop()
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
            node.left, node.right = node.right, node.left
        return root

    def preorder(self, root: 'Node') -> List[int]:
        """ 589.N叉树的前序遍历
            590.N叉树的后序遍历 589.稍作修改 """
        # 递归先序遍历
        res = []

        def backtracking(node):
            # 终止条件
            if not root:
                return
            # 单层递归
            res.append(node.val)
            for child in node.children:
                backtracking(child)
        backtracking(root)
        return res

    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        """ 101.对称二叉树
            同时遍历2棵树 """
        # 迭代 先序遍历
        # stack = [[root.left, root.right]]
        # while stack:
        #     left, right = stack.pop()
        #     if not (left or right):
        #         continue
        #     elif not (left and right):
        #         return False
        #     elif left.val != right.val:
        #         return False
        #     else:
        #         stack.extend([[left.left, right.right], [left.right, right.left]])
        # return True

        # 递归 先序遍历；《代码随想录》称之为“后序遍历”
        def backtracking(left, right):
            # 终止条件
            if not (left or right):
                return True
            elif not (left and right):
                return False
            elif left.val != right.val:
                return False
            else:
                # 单层递归，只有2节点均不空且值相等时才进入递归
                outside_res = backtracking(left.left, right.right)      # 这两行第1个参数--左右中；第2个参数--右左中。故《代码随想录》称之为“后序遍历”
                inner_res = backtracking(left.right, right.left)
                return outside_res and inner_res

        return backtracking(root.left, root.right)

    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        """ 100.相同的树 """
        # 迭代 先序遍历
        # if not (p or q):
        #     return True
        # elif not (p and q):
        #     return False
        # stack = [[p, q]]
        # while stack:
        #     left, right = stack.pop()
        #     if not (left or right):
        #         continue
        #     elif not (left and right):
        #         return False
        #     elif left.val != right.val:
        #         return False
        #     else:
        #         stack.extend([[left.left, right.left], [left.right, right.right]])
        # return True

        # 递归 先序遍历
        def backtracking(left, right):
            # 终止条件
            if not (left or right):
                return True
            elif not (left and right):
                return False
            elif left.val != right.val:
                return False
            else:
                outside_res = backtracking(left.left, right.left)
                inner_res = backtracking(left.right, right.right)
                return outside_res and inner_res

        return backtracking(p, q)

    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        """ 572.另一棵树的子树 """
        def backtracking(p, q):
            # 终止条件
            if not (p or q):
                return True
            elif not (p and q):
                return False
            elif p.val != q.val:
                return False
            else:
                return backtracking(p.left, q.left) and backtracking(p.right, q.right)

        # 遍历root，依次与subRoot比较
        stack = [root]
        while stack:
            node = stack.pop()
            if backtracking(node, subRoot):
                return True
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return False

if __name__ == "__main__":
    pass

"""
递归的实现：将函数参数值、局部变量、返回地址压入stack中，递归返回，即从stack顶弹出上一次递归的各项参数
"""