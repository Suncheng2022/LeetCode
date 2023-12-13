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


# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


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

        # 迭代 再写一遍
        # 操作两棵树，一般迭代法使用层序遍历比较好理解
        # queue = [[root.left, root.right]]
        # while queue:
        #     left, right = queue.pop(0)
        #     if not (left or right):
        #         continue
        #     elif not (left and right):
        #         return False
        #     elif left.val != right.val:
        #         return False
        #     else:                           # 均有值，且值相等
        #         queue.extend([[left.left, right.right], [left.right, right.left]])
        # return True

        # 递归 先序遍历；《代码随想录》称之为“后序遍历”
        # def backtracking(left, right):
        #     # 终止条件
        #     if not (left or right):
        #         return True
        #     elif not (left and right):
        #         return False
        #     elif left.val != right.val:
        #         return False
        #     else:
        #         # 单层递归，只有2节点均不空且值相等时才进入递归
        #         outside_res = backtracking(left.left, right.right)      # 这两行第1个参数--左右中；第2个参数--右左中。故《代码随想录》称之为“后序遍历”
        #         inner_res = backtracking(left.right, right.left)
        #         return outside_res and inner_res
        #
        # return backtracking(root.left, root.right)

        # 再写一遍 递归
        def backtracking(left, right):
            # 终止条件
            if not (left or right):
                return True
            elif not (left and right):
                return False
            elif left.val != right.val:
                return False
            else:
                outer_res = backtracking(left.left, right.right)
                inner_res = backtracking(left.right, right.left)
                return outer_res and inner_res

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
        """ 404.左叶子之和
            尝试过写 递归先序遍历，写不出来，因为传入的node只能用来判断子节点是否为左叶子，不能实现左右递归 """
        # 《代码随想录》迭代法，前中后序均可
        # 迭代前序遍历
        # stack = [root]
        # res = 0
        # while stack:
        #     node = stack.pop()
        #     if node.left and not (node.left.left or node.left.right):
        #         res += node.left.val
        #     if node.left:
        #         stack.append(node.left)
        #     if node.right:
        #         stack.append(node.right)
        # return res

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

        # 再写一遍 迭代遍历，前中后序均可
        stack = [root]          # 题目说明至少1个节点
        res = 0
        while stack:
            node = stack.pop()
            if node.left and not (node.left.left or node.left.right):
                res += node.left.val
            if node.right:      # 虽是前序遍历，本题入栈顺序也无所谓啦
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return res

        # 再写一遍 递归后序遍历
        # def backtracking(node):
        #     # 终止条件
        #     if not node:
        #         return 0
        #     elif not (node.left or node.right):     # 只有node为父节点，才可能判断出左叶子结点
        #         return 0
        #     # 单层递归
        #     left_val = backtracking(node.left)
        #     if node.left and not (node.left.left or node.left.right):       # node.left刚好是左叶子结点。node.right不可能是左叶子结点的。
        #         left_val = node.left.val
        #     right_val = backtracking(node.right)
        #     return left_val + right_val
        #
        # return backtracking(root)

    def findBottomLeftValue(self, root: Optional[TreeNode]) -> int:
        """ 513.找树左下角的值 """
        # 无脑 层序遍历
        # queue = [root]      # 题目说明至少1个节点
        # while queue:
        #     length = len(queue)
        #     tmp = []
        #     for _ in range(length):
        #         node = queue.pop(0)
        #         tmp.append(node.val)
        #         if node.left:
        #             queue.append(node.left)
        #         if node.right:
        #             queue.append(node.right)
        # return tmp[0]

        # 《代码随想录》递归，遍历顺序似乎都可以
        # maxDepth = 0
        # res = None
        #
        # def backtracking(node, depth):
        #     # 终止条件 遇到孩子节点终止
        #     if not (node.left or node.right):
        #         nonlocal maxDepth, res
        #         if depth > maxDepth:
        #             maxDepth = depth
        #             res = node
        #         return
        #     # 单层递归
        #     if node.left:                               # 递归函数终止条件没有判空，所以不空才能进入递归。
        #         backtracking(node.left, depth + 1)      # 体现回溯
        #     if node.right:
        #         backtracking(node.right, depth + 1)
        #
        # backtracking(root, 1)
        # return res.val

        # 再写一遍 递归先序遍历
        maxDepth = 0
        res = None

        def backtracking(node, depth):
            """ 当前节点node所在深度为depth """
            # 终止条件。没判空节点，因为下面控制了只有非空节点才能进递归
            if not (node.left or node.right):
                nonlocal maxDepth, res
                if depth > maxDepth:
                    maxDepth = max(maxDepth, depth)
                    res = node
                return
            # 单层递归
            if node.left:
                backtracking(node.left, depth + 1)          # 体现回溯
            if node.right:
                backtracking(node.right, depth + 1)

        backtracking(root, 1)
        return res.val

    def hasCycle(self, head: Optional[ListNode]) -> bool:
        """ 141.环形链表 """
        # 参考提交记录 快慢指针
        slow = fast = head
        while fast:
            slow = slow.next
            fast = fast.next
            if fast:
                fast = fast.next
            if fast == slow:        # 如果有环，最终会相遇
                break
        if not fast:
            return False
        return True

    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        """ 112.路径总和 """
        # 《代码随想录》递归先序遍历，因为不处理中节点，先中后均可
        # def backtracking(node, count):
        #     """ 遍历到当前节点node 还需count值[已经减去了当前node.val] """
        #     # 终止条件
        #     if not (node.left or node.right) and count == 0:
        #         return True
        #     elif not (node.left or node.right):
        #         return False
        #     # 单层递归
        #     if node.left:
        #         if backtracking(node.left, count - node.left.val):      # 体现回溯
        #             return True
        #     if node.right:
        #         if backtracking(node.right, count - node.right.val):    # 体现回溯
        #             return True
        #     return False
        #
        # if not root:
        #     return False
        # return backtracking(root, targetSum - root.val)

        # 自己写个递归 牛逼！
        def backtracking(node, path):
            # 终止条件
            if not (node.left or node.right) and sum(path) == targetSum:
                return True
            # 单层递归
            if node.left:
                if backtracking(node.left, path + [node.left.val]):
                    return True
            if node.right:
                if backtracking(node.right, path + [node.right.val]):
                    return True
            return False

        if not root:
            return False
        return backtracking(root, [root.val])

        # 《代码随想录》迭代先序遍历
        # if not root:
        #     return False
        # stack = [[root, root.val]]
        # while stack:
        #     node, total = stack.pop()
        #     if total == targetSum and not (node.left or node.right):
        #         return True
        #     if node.right:
        #         stack.append([node.right, total + node.right.val])
        #     if node.left:
        #         stack.append([node.left, total + node.left.val])
        # return False

    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        """ 113.路径总和II """
        # 递归先序遍历
        res = []

        def backtracking(node, path):
            """ 遍历截止到node，还需count值
                path已包含node.val """
            # 终止条件
            if not (node.left or node.right) and sum(path) == targetSum:
                res.append(path[:])
                return
            # 单层递归
            if node.left:       # 这里也可以写到终止条件
                backtracking(node.left, path + [node.left.val])
            if node.right:
                backtracking(node.right, path + [node.right.val])

        if not root:
            return []
        backtracking(root, [root.val])
        return res

    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        """ 106.从中序与后序遍历序列构造二叉树 """
        # 递归
        def backtracking(inorder, postorder):
            if not (len(inorder)):
                return
            root = TreeNode(postorder[-1])
            root_inorder_ind = inorder.index(postorder[-1])
            root.left = backtracking(inorder[:root_inorder_ind], postorder[:root_inorder_ind])
            root.right = backtracking(inorder[root_inorder_ind + 1:], postorder[root_inorder_ind:-1])
            return root

        return backtracking(inorder, postorder)

    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        """ 105.从前序与中序遍历序列构造二叉树
            认真分析索引就能过 加油！"""
        # 递归
        def backtracking(preorder, inorder):
            # 终止条件
            if not preorder:
                return          # 其实就是None
            root = TreeNode(preorder[0])
            root_inorder_ind = inorder.index(preorder[0])
            root.left = backtracking(preorder[1:root_inorder_ind + 1], inorder[:root_inorder_ind])
            root.right = backtracking(preorder[root_inorder_ind + 1:], inorder[root_inorder_ind + 1:])
            return root
        return backtracking(preorder, inorder)

    def constructMaximumBinaryTree(self, nums: List[int]) -> Optional[TreeNode]:
        """ 654.最大二叉树 """
        # 顺便，就用题目当递归函数吧
        # 终止条件
        if not nums:
            return
        # 单层递归
        root = TreeNode(max(nums))
        root_ind = nums.index(max(nums))
        root.left = self.constructMaximumBinaryTree(nums[:root_ind])
        root.right = self.constructMaximumBinaryTree(nums[root_ind + 1:])
        return root

    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
        """ 617.合并二叉树
            一般操作两棵树，迭代法使用层序遍历 """
        # 《代码随想录》递归前序遍历  自己竟然没做出来...
        # def backtracking(node1, node2):
        #     # 终止条件
        #     if not node1:
        #         return node2
        #     elif not node2:
        #         return node1
        #     # 单层递归
        #     node1.val += node2.val
        #     node1.left = backtracking(node1.left, node2.left)
        #     node1.right = backtracking(node1.right, node2.right)
        #     return node1
        #
        # return backtracking(root1, root2)

        # 再写一遍 复习 递归先序遍历
        # def backtracking(node1, node2):
        #     """ 将 以node1为根节点的子树、以node2为根节点的子树 合并到以node1为根节点的子树上 """
        #     # 终止条件 处理有 空节点 的情况，包括两个都为空时
        #     if not node1:
        #         return node2
        #     elif not node2:
        #         return node1
        #     # 单层递归  上面处理了关于 空节点 的情况，所以这里不考虑空了
        #     node1.val += node2.val
        #     node1.left = backtracking(node1.left, node2.left)
        #     node1.right = backtracking(node1.right, node2.right)
        #     return node1        # 首尾呼应，返回 合并到以node1为根节点后的子树
        #
        # return backtracking(root1, root2)

        # 《代码随想录》迭代法 可参考 101.对称二叉树
        # if not root1:
        #     return root2
        # elif not root2:
        #     return root1
        # queue = [[root1, root2]]
        # while queue:
        #     node1, node2 = queue.pop(0)
        #     node1.val += node2.val
        #     if node1.left and node2.left:
        #         queue.append([node1.left, node2.left])
        #     if node1.right and node2.right:
        #         queue.append([node1.right, node2.right])
        #
        #     if not node1.left and node2.left:
        #         node1.left = node2.left
        #     if not node1.right and node2.right:
        #         node1.right = node2.right
        # return root1

        # 再写一遍 同时操作两棵树通常使用层序遍历 还是看了答案
        # 处理 空 的情况
        if not root1:
            return root2
        elif not root2:
            return root1
        # 进入队列的认为不空了；后面 空节点 不入队列
        queue = [[root1, root2]]
        while queue:
            length = len(queue)
            for _ in range(length):
                node1, node2 = queue.pop(0)
                node1.val += node2.val
                if node1.left and node2.left:       # 空节点不入队列
                    queue.append([node1.left, node2.left])
                if node1.right and node2.right:     # 空节点不入队列
                    queue.append([node1.right, node2.right])

                if not node1.left and node2.left:   # 处理 空 相关的，只处理node1空的情况，因为往node1上合并嘛
                    node1.left = node2.left
                if not node1.right and node2.right:
                    node1.right = node2.right
        return root1


    def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        """ 700.二叉搜索树中的搜索
            BST，就不要忘记有序，搜索是有方向滴 """
        # 自己写出了和答案一样的，牛！简单到痛哭流涕
        # while root:
        #     if root.val > val:
        #         root = root.left
        #     elif root.val < val:
        #         root = root.right
        #     else:
        #         return root
        # return None

        # 递归 上来不会写BST的递归搜索，那就想一下普通二叉树的递归搜索
        def backtracking(node):
            # 终止条件
            if not node:
                return
            if node.val == val:
                return node
            # 单层递归
            if node.val > val:
                return backtracking(node.left)
            elif node.val < val:
                return backtracking(node.right)

        return backtracking(root)

    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        """ 98.验证二叉搜索树
            竟然没想到，判断中序是否递增呀 """
        # 迭代中序遍历
        # stack = []
        # cur = root
        # res = []
        # while cur or stack:
        #     if cur:
        #         stack.append(cur)
        #         cur = cur.left
        #     else:
        #         cur = stack.pop()
        #         if res and res[-1] >= cur.val:
        #             return False
        #         res.append(cur.val)
        #         cur = cur.right
        # return True

        # 《代码随想录》递归中序遍历，这自己是一点没能想出来
        # max_val = float('-inf')
        #
        # def backtracking(node):
        #     """ 当前遍历到node节点时，是否仍为BST """
        #     # 终止条件
        #     if not node:
        #         return True
        #     # 单层递归 中序遍历
        #     left = backtracking(node.left)
        #     nonlocal max_val
        #     if node.val <= max_val:     # BST中序遍历严格递增
        #         return False
        #     else:
        #         max_val = node.val
        #     right = backtracking(node.right)
        #     return left and right
        #
        # return backtracking(root)

        # 再写一遍 递归中序遍历 判断 是否BST
        pre = None              # 前一个访问的节点

        def backtracking(node):
            """ 判断以node为根节点的子树是不是BST """
            # 终止条件
            if not node:        # 空节点也是BST
                return True
            # 单层递归
            left_res = backtracking(node.left)
            nonlocal pre
            if pre and pre.val >= node.val:
                return False
            pre = node
            right_res = backtracking(node.right)
            return left_res and right_res           # 判断node的 左子树、右子树 是不是BST

        return backtracking(root)

    def getMinimumDifference(self, root: Optional[TreeNode]) -> int:
        """ 530.二叉搜索树的最小绝对差
            BST中序遍历严格递增 """
        # 迭代中序遍历
        # pre记录node而不是node.val，否则pre=0时 if进不去，会计算错误
        # cur = root
        # stack = []
        # pre = None
        # minVal = float('inf')
        # while cur or stack:
        #     if cur:
        #         stack.append(cur)
        #         cur = cur.left
        #     else:
        #         node = stack.pop()
        #         if pre:
        #             minVal = min(minVal, node.val - pre.val)
        #         pre = node
        #         cur = node.right
        # return minVal

        # 递归中序遍历
        pre = None      # 记录上一个访问的节点
        res = float('inf')

        def backtracking(node):
            # 终止条件
            if not node:
                return
            # 单层递归
            backtracking(node.left)
            nonlocal pre, res
            if pre:
                res = min(res, node.val - pre.val)
            pre = node
            backtracking(node.right)

        backtracking(root)
        return res

    def findMode(self, root: Optional[TreeNode]) -> List[int]:
        """ 501.二叉搜索数中的众数
            BST中序遍历严格，则众数肯定连续出现 """
        # 迭代中序遍历
        # stack = []
        # cur = root
        # count = 0
        # maxCount = 0
        # res = []
        # pre = None
        # while cur or stack:
        #     if cur:
        #         stack.append(cur)
        #         cur = cur.left
        #     else:
        #         cur = stack.pop()
        #         if pre and pre.val == cur.val:
        #             count += 1
        #         else:
        #             count = 1
        #         pre = cur
        #         if count == maxCount:
        #             res.append(cur.val)
        #         elif count > maxCount:
        #             maxCount = count
        #             res.clear()
        #             res.append(cur.val)
        #         cur = cur.right
        # return res

        # 递归中序遍历
        count = 0
        maxCount = 0
        pre = None          # 记录上一个访问的节点，这样才能计数
        res = []

        def backtracking(node):
            # 终止条件
            if not node:
                return
            # 单层递归
            backtracking(node.left)
            nonlocal count, maxCount, pre
            # 计数，同样的节点出现了多少次
            if not pre:
                count = 1
            elif pre.val != node.val:
                count = 1
            else:
                count += 1
            pre = node
            # 是否添加到res中
            if count == maxCount:
                res.append(node.val)
            elif count > maxCount:
                maxCount = count
                res.clear()
                res.append(node.val)
            backtracking(node.right)

        backtracking(root)
        return res

    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        """ 236.二叉树的最近公共祖先
            能想到自底向上-->递归后序遍历 """
        # 想到递归后序遍历 还是得《代码随想录》
        # def backtracking(node):
        #     """ 当前遍历节点node为根节点的子树是否找到 p或q，并向上返回：空 代表 没找到，不空 代表 找到 """
        #     # 终止条件
        #     if node in [p, q] or not node:      # if node in [p, q]对应情况二，node正是p或q，直接向上返回；not node为终止条件
        #         return node
        #     # 单层递归
        #     left_res = backtracking(node.left)
        #     right_res = backtracking(node.right)
        #     if not left_res:
        #         return right_res
        #     elif not right_res:
        #         return left_res
        #     elif left_res and right_res:
        #         return node
        #
        # return backtracking(root)

        # 再写一遍《代码随想录》
        def backtracking(node):
            """ 以node为根节点的子树 是否找到了p或q
                找到p或q就将其返回，找不到返回None代表没找到 """
            # 终止条件
            if not node:            # 空节点，肯定没找到
                return None
            elif node in [p, q]:    # 对应题解情况二，当前节点就是p或q，直接将其返回
                return node
            # 单层递归 后序遍历嘛
            left_res = backtracking(node.left)
            right_res = backtracking(node.right)
            if not (left_res and right_res):            # 若node的左右子树只有一支有返回值，则向上返回
                return left_res if left_res else right_res
            elif left_res and right_res:                # 若node的左右子树都有返回值，说明左右都找到了，那node就是公共祖先；因为是自底向上找的，所以 最近的/深度最大的 公共节点肯定会先找到，然后一步一步往上返回
                return node
            elif not (left_res or right_res):           # 若node的左右子树都没有返回值，说明左右都没找到，意思就是node不是p或q的祖先节点，所以 返回空
                return None

        return backtracking(root)

    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        """ 235.二叉搜索树的最近公共祖先 """
        # 回忆昨天 236.二叉树的最近公共祖先 递归后序遍历--找公共祖先自然想到自下而上，回溯递归，后序左右中
        # def backtracking(node):
        #     """ 以node为根节点的子树的左右节点是否找到p或q，找到则返回上层告知找到，否则返回空 """
        #     # 终止条件
        #     if not node or node in [p, q]:      # node为空，返回 空 告知上层没找到；node就是p或q，将本身返回上层告知找到了
        #         return node
        #     # 单层递归
        #     left_res = backtracking(node.left)
        #     right_res = backtracking(node.right)
        #     if not left_res:                    # 左子树没有返回结果，就把右子树返回结果返回（也可能二者都没有结果）
        #         return right_res
        #     elif not right_res:                 # 右子树没有返回结果，就把左子树返回结果返回（也可能二者都没有结果）
        #         return left_res
        #     elif left_res and right_res:        # 左右子树都有返回结果，就是左右子树分别找到了p、q，则node就是最近公共祖先
        #         return node
        #
        # return backtracking(root)

        # 《代码随想录》利用BST特性，搜索方向 递归顺序均可，因为中节点不需要处理
        # def backtracking(node):
        #     """ 以node为根节点的子树上是否找到 祖先节点 """
        #     # 终止条件
        #     if not node:
        #         return None
        #     # 单层递归
        #     if node.val < min(p.val, q.val):        # 如果当前节点node.val比p和q的值都小，则向右去找，找到就返回，本题就是寻找一条边
        #         right = backtracking(node.right)
        #         if right:
        #             return right
        #     if node.val > max(p.val, q.val):        # 同上
        #         left = backtracking(node.left)
        #         if left:
        #             return left
        #     if min(p.val, q.val) <= node.val <= max(p.val, q.val):
        #         return node
        # return backtracking(root)

        # 迭代法 BST搜索的有序性
        while root:
            if root.val > max(p.val, q.val):
                root = root.left
            elif root.val < min(p.val, q.val):
                root = root.right
            elif min(p.val, q.val) <= root.val <= max(p.val, q.val):
                return root
        return None

    def insertIntoBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        """ 701.二叉搜索树中的插入操作 """
        # if not root:
        #     return TreeNode(val)
        # cur = root
        # while cur:
        #     pre = cur
        #     if cur.val < val:
        #         cur = cur.right
        #     elif cur.val > val:
        #         cur = cur.left
        #     if not cur:
        #         newNode = TreeNode(val)
        #         if pre.val > val:
        #             pre.left = newNode
        #         else:
        #             pre.right = newNode
        #         break
        # return root

        # 《代码随想录》递归
        def backtracking(node):
            """ 插入到 以node为根节点 的子树上，并返回根节点--通过向上返回来建立父子节点关系 """
            # 终止条件
            if not node:                # BST，遇到空节点了，就是要插入了
                return TreeNode(val)
            # 单层递归
            if node.val > val:
                node.left = backtracking(node.left)
            elif node.val < val:
                node.right = backtracking(node.right)
            return node

        return backtracking(root)

    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        """ 450.删除二叉搜索树中的节点 """
        # 递归后序遍历 通过返回值建立父子关系
        # 自己写的，实现得好复杂--其实本来就很复杂，自己能写出来也可以
        # def backtracking(node):
        #     """ node是否为要删除的节点，是：向上层返回删除后结构 不是：返回本身 """
        #     # 终止条件
        #     if not node:
        #         return
        #     # 单层递归
        #     node.left = backtracking(node.left)
        #     node.right = backtracking(node.right)
        #     if node.val == key:
        #         if not node.right:
        #             node.right = node.left
        #             return node.right
        #         left_of_right = node.right
        #         while left_of_right.left:
        #             left_of_right = left_of_right.left
        #         left_of_right.left = node.left
        #         return node.right
        #     else:
        #         return node
        #
        # return backtracking(root)

        # 《代码随想录》递归 通过返回值确定新的父子关系，达到删除节点的目的
        def backtracking(node):
            """ node是否为要删除的节点，向上返回删除后的结构 分析情况1~5 """
            # 终止条件
            if not node:                                # 情况一：没找到要删除的节点
                return None
            if node.val == key:
                if not (node.left or node.right):       # 情况二：node为叶子节点
                    return None
                elif not node.left and node.right:      # 情况三：node只有右孩子节点
                    return node.right
                elif node.left and not node.right:      # 情况四：node只有左孩子节点
                    return node.left
                elif node.left and node.right:          # 情况五：node有左右孩子节点
                    left_of_right = node.right
                    while left_of_right.left:
                        left_of_right = left_of_right.left
                    left_of_right.left = node.left
                    return node.right
            # 单层递归
            node.left = backtracking(node.left)
            node.right = backtracking(node.right)
            return node

        return backtracking(root)

    def trimBST(self, root: Optional[TreeNode], low: int, high: int) -> Optional[TreeNode]:
        """ 669.修剪二叉搜索树 """
        # 递归 通过向上返回达到修剪的目的
        # 普通二叉树 的修剪，没有利用BST特性！
        # def backtracking(node):
        #     # 终止条件
        #     if not node:                        # 没找到要删除的节点
        #         return
        #     # 单层递归 后序遍历才行，前序不行
        #     node.left = backtracking(node.left)
        #     node.right = backtracking(node.right)
        #     if not low <= node.val <= high:     # 找到要删除的节点
        #         if not (node.left or node.right):
        #             return None
        #         elif not node.left and node.right:
        #             return node.right
        #         elif node.left and not node.right:
        #             return node.left
        #         elif node.left and node.right:
        #             tmp = node.right
        #             while tmp.left:
        #                 tmp = tmp.left
        #             tmp.left = node.left
        #             return node.right
        #     return node
        #
        # return backtracking(root)

        # 《代码随想录》递归 利用BST特性
        def backtracking(node):
            """ 返回以node为根节点的子树 删除后的 新根节点 """
            # 终止条件
            if not node:
                return
            # 单层递归
            if node.val < low:              # 当前节点node.val<low，则node及左孩子节点都不能要了
                return backtracking(node.right)
            if node.val > high:             # 同上
                return backtracking(node.left)
            node.left = backtracking(node.left)
            node.right = backtracking(node.right)
            return node

        return backtracking(root)

    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        """ 108.将有序数组转化为二叉搜索树 """
        # 《代码随想录》有序数组构建BST，取中间的做根节点，随便取也没意义
        def backtracking(start, end):
            # 终止条件
            if start > end:
                return None
            elif start == end:      # 这里其实不用写
                return TreeNode(nums[start])
            # 单层递归
            mid = (start + end) // 2
            node = TreeNode(nums[mid])
            node.left = backtracking(start, mid - 1)
            node.right = backtracking(mid + 1, end)
            return node

        return backtracking(0, len(nums) - 1)

    def convertBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """ 538.把二叉搜索树转化为累加树 """
        # 右中左递归遍历 自己写的！
        # pre = None
        #
        # def backtracking(node):
        #     # 终止条件
        #     if not node:
        #         return
        #     # 单层递归 右中左
        #     backtracking(node.right)
        #     nonlocal pre
        #     if pre:
        #         node.val += pre.val
        #     pre = node
        #     backtracking(node.left)
        #
        # backtracking(root)
        # return root

        # 迭代法 那就是中序遍历模板题了
        cur = root
        stack = []
        preVal = 0
        while cur or stack:
            if cur:
                stack.append(cur)
                cur = cur.right         # 注意本题是 右中左
            else:
                node = stack.pop()
                node.val += preVal
                preVal = node.val
                cur = node.left         # 注意本题是 右中左
        return root

if __name__ == "__main__":
    pass

"""
递归的实现：将函数参数值、局部变量、返回地址压入stack中，递归返回，即从stack顶弹出上一次递归的各项参数
"""