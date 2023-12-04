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

    # ------------------------- 以后从这里开始 -------------------------------------
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

    def maxDepth(self, root: Optional[TreeNode]) -> int:
        """ 104.二叉树的最大深度 """
        # 《代码随想录》递归 后序遍历求高度，根节点高度即二叉树的最大深度
        # def backtracking(node):
        #     """ 返回以node为根节点的二叉树的高度 """
        #     # 终止条件
        #     if not node:        # 遇到空节点，高度为0
        #         return 0
        #     # 单层递归  后序，左右中
        #     left_height = backtracking(node.left)
        #     right_height = backtracking(node.right)
        #     return 1 + max(left_height, right_height)
        #
        # return backtracking(root)

        # 《代码随想录》递归 前序遍历，前序求的是深度，从根节点开始的最大深度即为所求
        maxDepth = 0

        def backtracking(node, depth):
            """ 当前节点node为止，深度depth是多少；
                当前节点node能往下继续遍历，深度就+1 注意回溯操作 完美 """
            nonlocal maxDepth
            maxDepth = max(maxDepth, depth)
            # 终止条件
            if not (node.left or node.right):
                return
            # 单层递归
            if node.left:
                depth += 1
                backtracking(node.left, depth)
                depth -= 1
            if node.right:
                depth += 1
                backtracking(node.right, depth)
                depth -= 1

        if not root:
            return 0

        backtracking(root, 1)
        return maxDepth

    def maxDepth(self, root: 'Node') -> int:
        """ 559.N叉树的最大深度 """
        # 递归 先序遍历 试试
        maxDepth = 0

        def backtracking(node, depth):
            """ 当前节点的深度是depth，要注意回溯 """
            # 终止条件
            nonlocal maxDepth
            maxDepth = max(maxDepth, depth)
            # 单层递归
            for child in node.children:
                if child:
                    depth += 1
                    backtracking(child, depth)
                    depth -= 1

        if not root:
            return 0
        backtracking(root, 1)
        return maxDepth

    def minDepth(self, root: Optional[TreeNode]) -> int:
        """ 111.二叉树的最小深度 """
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

        #《代码随想录》递归后序遍历
        def backtracking(node):
            # 终止条件
            if not node:
                return 0
            # 单层递归
            left_res = backtracking(node.left)
            right_res = backtracking(node.right)
            if node.left and not node.right:
                return 1 + left_res
            elif not node.left and node.right:
                return 1 + right_res
            return 1 + min(left_res, right_res)

        return backtracking(root)

        # 《代码随想录》递归先序遍历
        # 自己尝试实现，失败了，还是得答案
        # minDepth = float('inf')
        #
        # def backtracking(node, depth):
        #     # 终止条件, 省略
        #     if not (node.left or node.right):
        #         nonlocal minDepth
        #         minDepth = min(minDepth, depth)
        #         return
        #     # 单层递归
        #     if node.left:
        #         backtracking(node.left, depth + 1)      # 体现回溯
        #     if node.right:
        #         backtracking(node.right, depth + 1)
        #
        # if not root:
        #     return 0
        # backtracking(root, 1)
        # return minDepth if minDepth != float('inf') else 0

    def countNodes(self, root: Optional[TreeNode]) -> int:
        """ 222.完全二叉树的节点个数 """
        # 《代码随想录》递归后序遍历 利用完全二叉树
        def backtracking(node):
            # 终止条件 要判断当前节点node为根的树是不是满二叉树，并返回节点数量
            if not node:
                return 0
            left, right = node.left, node.right
            depth_of_left, depth_of_right = 0, 0
            while left:             # 一直往左子树深入，看看深度为多少
                depth_of_left += 1
                left = left.left
            while right:            # 一直往右子树深入，看看深度为多少
                depth_of_right += 1
                right = right.right
            if depth_of_left == depth_of_right:
                return 2 ** (depth_of_left + 1) - 1       # 深度为depth的满二叉树节点数量的计算公式
            # 单层递归
            num_of_left = backtracking(node.left)
            num_of_right = backtracking(node.right)
            return 1 + num_of_left + num_of_right

        return backtracking(root)

        # 《代码随想录》递归后序遍历 普通二叉树
        # def backtracking(node):
        #     """ 截止到当前节点node所在位置，节点数量有多少 """
        #     # 终止条件
        #     if not node:
        #         return 0
        #     # 单层递归
        #     left_num = backtracking(node.left)
        #     right_num = backtracking(node.right)
        #     return 1 + max(left_num, right_num)
        #
        # if not root:
        #     return 0
        # return backtracking(root)

        # 递归先序遍历 自己写的
        # res = 0
        #
        # def backtracking(node):
        #     # 终止条件
        #     if not node:
        #         return
        #     # 单层递归
        #     nonlocal res
        #     res += 1
        #     backtracking(node.left)
        #     backtracking(node.right)
        #
        # backtracking(root)
        # return res

    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        """ 110.平衡二叉树
            求深度适用前序遍历，求高度适用后序遍历 """
        # 《代码随想录》后序递归遍历 求 左右子树的高度
        # def backtracking(node):
        #     """ 返回以当前节点node为根节点的子树的高度 -1表示已经不是平衡二叉树了 """
        #     # 终止条件
        #     if not node:
        #         return 0
        #     # 单层递归 后序遍历
        #     left_height = backtracking(node.left)
        #     if left_height == -1:
        #         return -1
        #     right_height = backtracking(node.right)
        #     if right_height == -1:
        #         return -1
        #     if abs(left_height - right_height) <= 1:
        #         return 1 + max(left_height, right_height)
        #     else:
        #         return -1
        #
        # return False if backtracking(root) == -1 else True

        # 《代码随想录》再写一遍 后序递归遍历，求左、右子树高度嘛
        def backtracking(node):
            """ 返回以node为根节点的子树的高度 -1表示已经不是二叉平衡树了 """
            # 终止条件
            if not node:
                return 0
            # 单层递归 后序遍历
            left_height = backtracking(node.left)
            if left_height == -1:       # 当前节点node的左子树已经不是平衡二叉树了，则当前节点node的子树也不是了
                return -1
            right_height = backtracking(node.right)
            if right_height == -1:      # 当前节点node的右子树已经不是二叉平衡树了，则当前节点node的子树也不是了
                return -1
            if abs(left_height - right_height) <= 1:        # 若左右子树高度相差<=1，是二叉平衡树，返回二叉平衡树的高度
                return 1 + max(left_height, right_height)
            else:
                return -1                                   # 若左右子树高度相差>1，不是二叉平衡树，返回-1告诉上一层

        return True if backtracking(root) != -1 else False

    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
        """ 257.二叉树的所有路径
            二叉树中第一次使用回溯 """
        # 《代码随想录》求路径，前序遍历比较方便，方便记录父子节点关系
        res = []
        path = []

        def backtracking(node):
            path.append(node.val)                   # 中，放在这，因为要确保所有节点都能进到path
            # 终止条件
            if not (node.left or node.right):       # 遇到叶节点终止
                res.append("->".join([str(val) for val in path]))
                return
            # 单层递归
            if node.left:
                backtracking(node.left)
                path.pop()      # 求所有路径，必须回溯，这样才能回退一个路径、进入下一个路径
            if node.right:
                backtracking(node.right)
                path.pop()      # 同上

        backtracking(root)
        return res

    def sumOfLeftLeaves(self, root: Optional[TreeNode]) -> int:
        """ 404.左叶子之和 """
        # 《代码随想录》迭代法，前中后序均可
        # 迭代前序遍历
        stack = [root]
        res = 0
        while stack:
            node = stack.pop()
            if node.left and not (node.left.left or node.left.right):
                res += node.left.val
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
        return res

        # 《代码随想录》并参考11.8记录 递归后序遍历
        # def backtracking(node):
        #     """ 以node为根节点的树 的 左叶子节点之和 """
        #     # 终止条件
        #     if not node:
        #         return 0
        #     elif not (node.left or node.right):
        #         return 0
        #     # 单层递归
        #     leftVal = backtracking(node.left)
        #     if node.left and not (node.left.left or node.left.right):       # 当前节点node的左节点 可能就是 左叶子
        #         leftVal = node.left.val
        #     rightVal = backtracking(node.right)
        #     return leftVal + rightVal
        #
        # return backtracking(root)


if __name__ == "__main__":
    pass

"""
递归的实现：将函数参数值、局部变量、返回地址压入stack中，递归返回，即从stack顶弹出上一次递归的各项参数
"""