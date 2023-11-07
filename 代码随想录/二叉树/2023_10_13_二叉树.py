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

    def maxDepth(self, root: 'Node') -> int:
        """ 559.N叉树的最大深度
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
                queue.extend(node.children)
            if tmp:
                res.append(tmp)
        return len(res)

    def minDepth(self, root: Optional[TreeNode]) -> int:
        """ 111.二叉树的最小深度
            简单 """
        if not root:
            return 0
        queue = [root]
        res = []
        while queue:
            length = len(queue)
            tmp = []
            for _ in range(length):
                node = queue.pop(0)
                tmp.append(node.val)
                if not (node.left or node.right):       # node没有左右孩子
                    return len(res) + 1 if tmp else len(res)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            if tmp:
                res.append(tmp)
        return len(res)

    def countNodes(self, root: Optional[TreeNode]) -> int:
        """ 222.完全二叉树的节点个数 """
        # 《代码随想录》递归写法 效率高很多
        # 递归三部曲 1.递归函数的 参数、返回值 2.终止条件 3.单层搜索
        def backtracking(node):
            # 终止条件
            if not node:
                return 0
            # 单层搜索
            leftNum = backtracking(node.left)
            rightNum = backtracking(node.right)
            return leftNum + rightNum + 1

        return backtracking(root)

        # 自己写的 层序遍历 性能较差
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
        # return sum([len(ls) for ls in res])

    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        """ 110.平衡二叉树
            简单 """
        # 《代码随想录》 递归
        def backtracking(node):
            """ 返回树的高度；
                或返回-1 """
            # 终止条件
            if not node:
                return 0
            leftNum = backtracking(node.left)
            if leftNum == -1:
                return -1
            rightNum = backtracking(node.right)
            if rightNum == -1:
                return -1
            if abs(leftNum - rightNum) > 1:
                return -1
            return 1 + max(leftNum, rightNum)

        res = backtracking(root)
        return True if res != -1 else False

    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
        """ 257.二叉树的所有路径
            简单 本题，首次二叉树递归/回溯  递归、回溯 本一家 """
        # 《代码随想录》递归
        path = []
        res = []

        def backtracking(node):
            path.append(node.val)
            # 终止条件
            if not (node.left or node.right):      # node不为空，且没有孩子节点--即 叶子结点
                res.append("->".join([str(c) for c in path]))
                return
            # 单层搜索
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
            简单
            难点在如何判断是左叶子 """
        # 迭代法 递归不好理解
        stack = [root]
        res = 0
        while stack:
            node = stack.pop()
            if node.left and not (node.left.left or node.left.right):
                res += node.left.val
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return res

    def findBottomLeftValue(self, root: Optional[TreeNode]) -> int:
        """ 513.找树左下角的值 """
        queue = [root]
        res = []
        while queue:
            length = len(queue)
            tmp = []
            for _ in range(length):
                node = queue.pop(0)
                if not node:
                    continue
                tmp.append(node)
                queue.extend([node.left, node.right])
            if tmp:
                res.append(tmp)
        return res[-1][0].val

    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        """ 112.路径总和 """
        # def backtracking(node, count):
        #     # 终止条件
        #     if not (node.left or node.right) and count == 0:        # 找到叶子节点，且路径和恰为目标和
        #         return True
        #     if not (node.left or node.right):                       # 找到叶子节点，但不符合题意
        #         return False
        #     # 单层搜索
        #     if node.left:
        #         if backtracking(node.left, count - node.left.val):
        #             return True
        #     if node.right:
        #         if backtracking(node.right, count - node.right.val):
        #             return True
        #     return False
        #
        # if not root:
        #     return False
        # return backtracking(root, targetSum - root.val)

        # 再写一遍 递归
        # def backtracking(node, count):
        #     # 终止条件
        #     if not (node.left or node.right) and count == 0:
        #         return True
        #     if not (node.left or node.right):
        #         return False
        #     # 单层搜索
        #     if node.left:
        #         if backtracking(node.left, count - node.left.val):
        #             return True
        #     if node.right:
        #         if backtracking(node.right, count - node.right.val):
        #             return True
        #     return False
        #
        # if not root:
        #     return False
        # return backtracking(root, targetSum - root.val)

        # 迭代 由先序遍历迭代修改而来
        if not root:
            return False
        stack = [[root, root.val]]
        while stack:
            node, val = stack.pop()
            if not (node.left or node.right) and val == targetSum:
                return True
            if node.right:
                stack.append([node.right, val + node.right.val])
            if node.left:
                stack.append([node.left, val + node.left.val])
        return False

    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        """ 113.路径总和II """
        # 递归
        res = []
        path = []

        def backtracking(node, count):
            if not (node.left or node.right) and count == 0:
                res.append(path[:])
                return
            if node.left:
                path.append(node.left.val)
                backtracking(node.left, count - node.left.val)
                path.pop()
            if node.right:
                path.append(node.right.val)
                backtracking(node.right, count - node.right.val)
                path.pop()

        if not root:
            return res
        path.append(root.val)
        backtracking(root, targetSum - root.val)
        return res

    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        """ 106.从中序与后序遍历序列构造二叉树 """
        # 直接改为递归
        # if not (inorder and postorder):
        #     return None
        # inorderRootVal = postorder[-1]
        # root = TreeNode(inorderRootVal)
        #
        # if len(inorder) == 1:
        #     return root
        #
        # inorderRootInd = inorder.index(inorderRootVal)
        # leftInorder = inorder[:inorderRootInd]
        # rightInorder = inorder[inorderRootInd + 1:]
        #
        # leftPostorder = postorder[:len(leftInorder)]
        # rightPostorder = postorder[len(leftPostorder):-1]
        #
        # root.left = self.buildTree(leftInorder, leftPostorder)
        # root.right = self.buildTree(rightInorder, rightPostorder)
        #
        # return root

        # 再写一遍
        if not (inorder and postorder):
            return None

        rootVal = postorder[-1]
        root = TreeNode(rootVal)

        if len(postorder) == 1:
            return root

        rootInorderInd = inorder.index(rootVal)
        leftInorder = inorder[:rootInorderInd]
        rightInorder = inorder[rootInorderInd + 1:]

        leftPostorder = postorder[:len(leftInorder)]
        rightPostorder = postorder[len(leftInorder):-1]

        root.left = self.buildTree(leftInorder, leftPostorder)
        root.right = self.buildTree(rightInorder, rightPostorder)

        return root

    def constructMaximumBinaryTree(self, nums: List[int]) -> Optional[TreeNode]:
        """ 654.最大二叉树
            构造二叉树，一般前序遍历，先构造根节点、再构造左右 """
        maxVal = max(nums)
        maxInd = nums.index(maxVal)
        root = TreeNode(maxVal)
        if len(nums) == 1:
            return root
        if maxInd > 0:
            root.left = self.constructMaximumBinaryTree(nums[:maxInd])
        if maxInd < len(nums) - 1:
            root.right = self.constructMaximumBinaryTree(nums[maxInd + 1:])
        return root

    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
        """ 617.合并二叉树 """
        # 递归，不好懂
        # if not root1:
        #     return root2
        # if not root2:
        #     return root1
        # root1.val += root2.val
        # root1.left = self.mergeTrees(root1.left, root2.left)
        # root1.right = self.mergeTrees(root1.right, root2.right)
        # return root1

        # 迭代 用队列模拟层序遍历
        if not root1:
            return root2
        if not root2:
            return root1
        queue = [[root1, root2]]
        while queue:
            node1, node2 = queue.pop(0)
            node1.val += node2.val
            if node1.left and node2.left:
                queue.append([node1.left, node2.left])
            if node1.right and node2.right:
                queue.append([node1.right, node2.right])
            if not node1.left and node2.left:
                node1.left = node2.left
            if not node1.right and node2.right:
                node1.right = node2.right
        return root1

    def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        """ 700.二叉搜索树中的搜索 """
        # 尝试普通递归
        # self.res = None
        #
        # def backtracking(node):
        #     if not node:
        #         return
        #     if node.val == val:
        #         self.res = node
        #     backtracking(node.left)
        #     backtracking(node.right)
        #
        # backtracking(root)
        # return self.res

        # 《代码随想录》毕竟是BST嘛，要和普通二叉树有所区别
        # def backtracking(node):
        #     # 终止条件
        #     if not node or node.val == val:
        #         return node
        #     # 单层搜索
        #     if node.val > val:      # 则搜索 左子树
        #         res = backtracking(node.left)
        #     if node.val < val:
        #         res = backtracking(node.right)
        #     return res if res else None
        #
        # return backtracking(root)

        # 《代码随想录》迭代法 感动到哭的简单
        while root:
            if root.val > val:
                root = root.left
            elif root.val < val:
                root = root.right
            else:
                return root
        return None

    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        """ 98.验证二叉搜索树
            BST的中序遍历递增，判断中序遍历是否递增即可 """
        # 递归 中序遍历
        # res = []
        #
        # def backtracking(node):
        #     # 终止条件
        #     if not node:
        #         return
        #     # 单层搜索
        #     backtracking(node.left)
        #     res.append(node.val)
        #     backtracking(node.right)
        #
        # backtracking(root)
        # for i in range(len(res) - 1):       # 这里和《代码随想录》中答案出奇地相同
        #     if res[i] >= res[i + 1]:
        #         return False
        # return True

        # 迭代 中序遍历，不复习 忘了吧
        res = []
        stack = []
        cur = root
        while cur or stack:
            if cur:
                stack.append(cur)
                cur = cur.left
            else:
                cur = stack.pop()
                if res and res[-1] >= cur.val:
                    return False
                res.append(cur.val)
                cur = cur.right
        return True

    def getMinimumDifference(self, root: Optional[TreeNode]) -> int:
        """ 530.二叉搜索树的最小绝对差 """
        # 迭代法 【中序遍历 迭代法，不同于 前/后序遍历】
        # res = []
        # stack = []
        # cur = root
        # minSubVal = float('inf')
        # while cur or stack:
        #     if cur:
        #         stack.append(cur)
        #         cur = cur.left
        #     else:
        #         cur = stack.pop()
        #         if res:
        #             minSubVal = min(minSubVal, cur.val - res[-1])
        #         res.append(cur.val)
        #         cur = cur.right
        # return minSubVal

        # 递归 中序遍历，BST嘛
        res = []
        self.minSubVal = float('inf')

        def backtracking(node):
            # 终止条件
            if not node:
                return
            # 单层搜索
            backtracking(node.left)
            if res:
                self.minSubVal = min(self.minSubVal, node.val - res[-1])
            res.append(node.val)
            backtracking(node.right)

        backtracking(root)
        return self.minSubVal

    def findMode(self, root: Optional[TreeNode]) -> List[int]:
        """ 501.二叉搜索数中的众数
            这道题BST出现重复元素了 """
        # 迭代法 中序遍历 感觉不太流畅
        # 没有考虑BST特性--中序递增
        # from collections import Counter
        #
        # res = []
        # stack = []
        # cur = root
        # while cur or stack:
        #     if cur:
        #         stack.append(cur)
        #         cur = cur.left
        #     else:
        #         cur = stack.pop()
        #         res.append(cur.val)
        #         cur = cur.right
        # v2c = Counter(res)
        # v2c = [[k, v] for k, v in v2c.items()]
        # v2c.sort(key=lambda x: x[1], reverse=True)
        # res = [v2c[0][0]]
        # for k, v in v2c[1:]:        # 和答案有点相似哦
        #     if v < v2c[0][1]:
        #         break
        #     res.append(k)
        # return res

        # 看下答案的 '一次遍历' 很麻烦，各种判断
        # 号称是根据BST性质来的
        self.pre = None
        self.count = 0
        self.maxCount = 0
        self.res = []

        def backtracking(node):
            # 终止条件
            if not node:
                return
            # 单层遍历
            backtracking(node.left)
            # 处理 中 节点
            if not self.pre:
                self.count = 1
            elif self.pre.val == node.val:
                self.count += 1
            else:
                self.count = 1
            self.pre = node
            if self.count == self.maxCount:
                self.res.append(node.val)
            elif self.count > self.maxCount:
                self.maxCount = self.count
                self.res = [node.val]
            backtracking(node.right)

        backtracking(root)
        return self.res

    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        """ 236.二叉树的最近公共祖先
            不太懂，答案很不好懂 """
        # 终止条件
        if root == p or root == q or not root:      # 情况二
            return root
        # 单层搜索，后序遍历 左右中
        # 遍历整棵树，用变量接住
        left = self.lowestCommonAncestor(root.left, p, q)       # 递归返回值不为空，说明找到了p或q
        right = self.lowestCommonAncestor(root.right, p, q)     # 同上，不可能左右子树找到同一节点，题目已说明无重复节点
        if left and right:                          # 情况一
            return root

        if left and not right:
            return left
        elif not left and right:
            return right
        else:
            return None

    def lowestCommonAncestor_BST(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        """ 235.二叉搜索树的最近公共祖先
            BST，则利用其有序的特点；同 236.二叉树的最近公共祖先 相似 """
        # 终止条件
        # if not root:
        #     return root
        # # 单层搜索  顺序无所谓，因为不需要特别处理 中 节点
        # if root.val > p.val and root.val > q.val:
        #     left = self.lowestCommonAncestor_BST(root.left, p, q)
        #     if left:
        #         return left
        # elif root.val < p.val and root.val < q.val:
        #     right = self.lowestCommonAncestor_BST(root.right, p, q)
        #     if right:
        #         return right
        # else:
        #     return root

        # 迭代法
        while root:
            if root.val > p.val and root.val > q.val:
                root = root.left
            elif root.val < p.val and root.val < q.val:
                root = root.right
            else:
                return root

    def insertIntoBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        """ 701.二叉搜索树中的插入操作 """
        # 终止条件
        # if not root:            # 遍历到空节点，插入并返回给上一层，以完成父子节点赋值操作
        #     return TreeNode(val)
        # # 单层搜索
        # if root.val < val:      # val大，就用root的右子树去接
        #     root.right = self.insertIntoBST(root.right, val)
        # elif root.val > val:    # val小，就用root的左子树去接
        #     root.left = self.insertIntoBST(root.left, val)
        # return root

        # 迭代法
        if not root:
            return TreeNode(val)
        cur = root
        parent = root
        while cur:
            parent = cur
            if cur.val < val:
                cur = cur.right
            elif cur.val > val:
                cur = cur.left
        node = TreeNode(val)
        if parent.val > val:
            parent.left = node
        else:
            parent.right = node
        return root

    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        """ 450.删除二叉搜索树中的节点 """
        # if not root:        # 遍历到空节点，返回空
        #     return None
        # if root.val == key:
        #     # 判断顺序需要注意，怕乱也可以写多个if判断
        #     if not (root.left or root.right):       # 情况一：删除的是叶子节点
        #         return None
        #     elif not root.right:                    # 情况二：有左孩子
        #         return root.left
        #     elif not root.left:                     # 情况三：有右孩子
        #         return root.right
        #     else:                                   # 情况四：有左、右孩子
        #         left_of_right = root.right
        #         while left_of_right.left:
        #             left_of_right = left_of_right.left
        #         left_of_right.left = root.left
        #         return root.right
        # if root.val > key:
        #     root.left = self.deleteNode(root.left, key)         # 承接 删除后补位上来的节点
        # elif root.val < key:
        #     root.right = self.deleteNode(root.right, key)       # 同上，承接
        # return root

        # 再写一遍 递归
        # 终止条件
        if not root:
            return None
        if root.val == key:
            if not (root.left or root.right):       # 情况一：叶子节点
                return None
            elif not root.left:                     # 情况二：没有左孩子，意味着可能有右孩子，那就返回右孩子 空就空吧；这里必须判 是否空，要考虑if判断顺序
                return root.right
            elif not root.right:                    # 情况三：没有右孩子
                return root.left
            else:                                   # 情况四：有左、右孩子. 把左孩子挂到右孩子的最左
                left_of_right = root.right
                while left_of_right.left:
                    left_of_right = left_of_right.left
                left_of_right.left = root.left
                return root.right
        # 单层搜索
        if root.val < key:                          # key大，用右子树来承接
            root.right = self.deleteNode(root.right, key)
        elif root.val > key:                        # key小，用左子树来承接
            root.left = self.deleteNode(root.left, key)
        return root

    def trimBST(self, root: Optional[TreeNode], low: int, high: int) -> Optional[TreeNode]:
        """ 669.修剪二叉搜索树 """
        if not root:
            return None
        if root.val < low:
            return self.trimBST(root.right, low, high)      # 寻找符合[low,high]闭区间的节点
        elif root.val > high:
            return self.trimBST(root.left, low, high)       # 同上
        root.left = self.trimBST(root.left, low, high)
        root.right = self.trimBST(root.right, low, high)
        return root

    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        """ 108.将有序数组转换为二叉搜索树 """
        def backtrakcing(leftInd, rightInd):
            if leftInd > rightInd:
                return None
            midInd = (leftInd + rightInd) // 2
            root = TreeNode(nums[midInd])
            root.left = backtrakcing(leftInd, midInd - 1)
            root.right = backtrakcing(midInd + 1, rightInd)
            return root
        return backtrakcing(0, len(nums) - 1)

    def convertBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """ 538.把二叉搜索树转换为累加树 """
        self.pre = 0

        def backtracking(node):
            if not node:
                return
            backtracking(node.right)
            node.val += self.pre
            self.pre = node.val
            backtracking(node.left)
        backtracking(root)
        return root

if __name__ == "__main__":
    sl = Solution()

    """
    递归三要素：
        1.递归函数的参数和返回值
        2.终止条件
        3.单层递归逻辑
    """