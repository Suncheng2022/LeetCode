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


# class Node:
#     def __init__(self, val=0, left=None, right=None, next=None):
#         self.val = val
#         self.left = left
#         self.right = right
#         self.next = next

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

    def largestValues(self, root: Optional[TreeNode]) -> List[int]:
        """ 515.在每个数行中找最大值
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
                res.append(max(tmp))
        return res

    def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
        """ 116.填充每个节点的下一个右侧节点指针
            中等 """
        queue = [root]
        res = []
        while queue:
            length = len(queue)
            tmp = []
            for i in range(length):
                node = queue.pop(0)
                if not node:
                    continue
                tmp.append(node)
                queue.extend([node.left, node.right])
            if tmp:
                res.append(tmp)
        for tmp in res:
            for i in range(len(tmp) - 1):
                tmp[i].next = tmp[i + 1]
        return root

    def connectII(self, root: 'Node') -> 'Node':
        """ 117.填充每个节点的下一个右侧节点指针II
            中等
            本题，空node不能进队列 """
        queue = []
        res = []
        if not root:
            return root
        queue.append(root)
        while queue:
            length = len(queue)
            tmp = []    # 空node不能进来
            for _ in range(length):
                node = queue.pop(0)
                if not node:
                    continue
                tmp.append(node)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(tmp)
        for tmp in res:
            for i in range(len(tmp) - 1):
                tmp[i].next = tmp[i + 1]
        return root

    def maxDepth(self, root: Optional[TreeNode]) -> int:
        """ 104.二叉树的最大深度
            简单
            稍作修改 灵活运用吧 """
        queue = [root]
        res = 0
        while queue:
            length = len(queue)
            tmp = 0
            for _ in range(length):
                node = queue.pop(0)
                if not node:
                    continue
                tmp += 1
                queue.extend([node.left, node.right])
            if tmp:
                res += 1
        return res

    def minDepth(self, root: Optional[TreeNode]) -> int:
        """ 111.二叉树的最小深度
            简单
            自己没想对... """
        # 自己想的有些复杂
        # queue = [root]
        # res = []
        # while queue:
        #     length = len(queue)
        #     tmp = []
        #     for _ in range(length):
        #         node = queue.pop(0)
        #         if not node:
        #             continue
        #         tmp.append(node)
        #         queue.extend([node.left, node.right])
        #         if not (node.left or node.right):
        #             break
        #     if tmp:
        #         res.append(tmp)
        # return len(res)

        # 《代码随想录》
        queue = []
        res = 0
        if not root:
            return 0
        queue.append(root)
        while queue:
            res += 1
            length = len(queue)
            # tmp = []
            for _ in range(length):
                node = queue.pop(0)
                if not (node.left or node.right):
                    return res
                # tmp.append(node)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return res

    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """ 226.翻转二叉树
            简单
            只要遍历的过程中，交换node的2个孩子节点就可以 """
        # 自己写的层序遍历
        # queue = []
        # if not root:
        #     return root
        # queue.append(root)
        # while queue:
        #     length = len(queue)
        #     for _ in range(length):
        #         node = queue.pop(0)
        #         if not node:
        #             continue
        #         node.left, node.right = node.right, node.left
        #         queue.extend([node.left, node.right])
        # return root

        # 《代码随想录》先序递归
        # def backtracking(node):
        #     if not node:
        #         return
        #     node.left, node.right = node.right, node.left
        #     backtracking(node.left)
        #     backtracking(node.right)
        #
        # backtracking(root)
        # return root

        # 《代码随想录》先序迭代
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
        """ 589.N叉树的前序遍历
            简单
            先序递归 """
        res = []

        def backtracking(node):
            if not node:
                return
            res.append(node.val)
            for c in node.children:
                backtracking(c)

        backtracking(root)
        return res

    def postorder(self, root: 'Node') -> List[int]:
        """ 590.N叉树的后序遍历
            简单
            后序递归 """
        res = []

        def backtracking(node):
            if not node:
                return
            for c in node.children:
                backtracking(c)
            res.append(node.val)

        backtracking(root)
        return res

    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        """ 101.对称二叉树
            简单 """
        # 递归解法 三部曲
        # def backtracking(left, right):
        #     # 处理 空 的情况
        #     if left and not right:
        #         return False
        #     elif not left and right:
        #         return False
        #     elif not left and not right:        # 左右都为 空，返回True
        #         return True
        #     # 处理 左右节点不等 的情况
        #     elif left.val != right.val:
        #         return False
        #     # 处理 左右均有节点 且 相等 的情况
        #     out = backtracking(left.left, right.right)      # 比较 左节点、右节点 的外侧
        #     inner = backtracking(left.right, right.left)    # 比较              的内侧
        #     return out and inner
        #
        # return backtracking(root.left, root.right)

        # 迭代法，但不是普通二叉树的那种迭代法
        queue = [root.left, root.right]      # 使用队列实现本题的迭代法
        while queue:
            left, right = queue.pop(0), queue.pop(0)
            if not (left or right):
                continue
            if not left or not right or left.val != right.val:
                return False
            if left.val == right.val:
                queue.extend([left.left, right.right, left.right, right.left])
        return True

    def maxDepth(self, root: Optional[TreeNode]) -> int:
        """ 104.二叉树的最大深度
            简单 """
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
        return len(res)

if __name__ == "__main__":
    sl = Solution()

    """
    递归三要素：
        1.递归函数的参数和返回值
        2.终止条件
        3.单层递归逻辑
    """