"""
2025.1.6    二叉树, 递归三要素:
                    1.确定递归函数的参数和返回值
                    2.确定终止条件
                    3.确定单层递归调用的逻辑(递归函数的实现)
"""
from typing import List, Optional


class TreeNode:
    """ 二叉树节点定义 """
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Node:
    """ N叉树节点定义 """
    def __init__(self, val: Optional[int] = None, children: Optional[List["Node"]] = None):
        self.val = val
        self.children = children

class Node_r:
    """ 节点右指针相关题目 """
    def __init__(self, val: Optional[int] = None, left: Optional['Node_r'] = None, right: Optional['Node_r'] = None, next: Optional['Node_r'] = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next

class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        """ 144.二叉树的前序遍历 """
        # 时间:O(n) 空间:O(n)
        def func(cur, res):
            if cur is None:
                return
            res.append(cur.val)
            func(cur.left, res)
            func(cur.right, res)
        res = []
        func(root, res)
        return res
    
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        """ 94.二叉树的中序遍历 """
        # 时间:O(n) 空间:O(n)
        def func(cur, res):
            if cur is None:             # None是python的单例对象, is判断二者是否为同一个对象, ==判断二者值是否相等. 与None的比较是为了比较身份而不是值, 所以推荐 is None
                return
            func(cur.left, res)
            res.append(cur.val)
            func(cur.right, res)
        res = []
        func(root, res)
        return res
    
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        """ 145.二叉树的后序遍历 """
        # 时间:O(n) 空间:O(n)
        def func(cur, res):
            if cur is None:
                return
            func(cur.left, res)
            func(cur.right, res)
            res.append(cur.val)
        res = []
        func(root, res)
        return res
    
    def preorderTraversal_iter(self, root: Optional[TreeNode]) -> List[int]:
        """ 二叉树迭代遍历_前序遍历 """

        # 再来一遍
        if root is None:
            return []
        stack = [root]
        res = []
        while len(stack):
            cur = stack.pop()
            res.append(cur.val)
            if cur.right:
                stack.append(cur.right)
            if cur.left:
                stack.append(cur.left)
        return res

        # stack = []
        # res = []
        # if root is None:
        #     return
        # stack.append(root)
        # while len(stack):
        #     cur = stack.pop()
        #     res.append(cur.val)
        #     if cur.right:
        #         stack.append(cur.right)
        #     if cur.left:
        #         stack.append(cur.left)
        # return res
    
    def inorderTraversal_iter(self, root: Optional[TreeNode]) -> List[int]:
        """ 二叉树迭代遍历_中序遍历 \n
            可不是前序遍历随便改改, 因为中序遍历的访问顺序和处理顺序不同 """
        
        # 再来一遍
        if root is None:
            return []
        res = []
        stack = []
        cur = root
        while cur or len(stack):
            if cur:
                stack.append(cur)
                cur = cur.left
            else:
                cur = stack.pop()
                res.append(cur.val)
                cur = cur.right
        return res

        # stack = []
        # res = []
        # cur = root
        # while cur or len(stack):
        #     if cur:
        #         stack.append(cur)
        #         cur = cur.left
        #     else:
        #         cur = stack.pop()
        #         res.append(cur.val)
        #         cur = cur.right
        # return res

    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        """ 二叉树迭代遍历_后序遍历 \n
            前序遍历 中左右 --> 中右左 --> 反转结果, 左右中 """
        # 再来一遍
        res = []
        if root is None:
            return res
        stack = [root]
        while len(stack):
            cur = stack.pop()
            res.append(cur.val)
            if cur.left:
                stack.append(cur.left)
            if cur.right:
                stack.append(cur.right)
        return res[::-1]
        
        # if root is None:
        #     return
        # res = []
        # stack = [root]
        # while len(stack):
        #     cur = stack.pop()
        #     res.append(cur.val)
        #     if cur.left:
        #         stack.append(cur.left)
        #     if cur.right:
        #         stack.append(cur.right)
        # return res[::-1]

    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        """ 二叉树层序遍历_打十个! \n
            层序遍历--队列 """
        # 时间:O(n^2) 空间:O(n)
        res = []
        if root is None:
            return res
        queue = [root]
        while len(queue):
            res_level = []
            n = len(queue)              # queue的长度在变化
            for _ in range(n):
                node = queue.pop(0)
                res_level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(res_level)
        return res
    
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        """ 102.二叉树的层序遍历 """
        # 时间:O(n^2), 其实O(n)也对 空间:O(n)
        res = []
        if root is None:
            return res
        queue = [root]
        while len(queue):
            n = len(queue)
            res_level = []
            for _ in range(n):
                node = queue.pop(0)
                res_level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(res_level)
        return res
    
    def levelOrderBottom(self, root: Optional[TreeNode]) -> List[List[int]]:
        """ 107.二叉树的层序遍历II """
        # 时间:O(n^2), 如果用deque则为O(n) 空间:O(n)
        # res = []
        # if root is None:
        #     return res
        # queue = [root]
        # while queue:
        #     n = len(queue)
        #     res_level = []
        #     for _ in range(n):
        #         node = queue.pop(0)
        #         res_level.append(node.val)
        #         if node.left:
        #             queue.append(node.left)
        #         if node.right:
        #             queue.append(node.right)
        #     res.append(res_level)
        # return res[::-1]

        # Python的deque
        # 时间:O(n) 因为用了deque 空间:O(n)
        from collections import deque
        
        res = []
        if root is None:
            return res
        queue = deque([root])
        while queue:
            res_level = []
            for _ in range(len(queue)):
                node = queue.popleft()
                res_level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(res_level)
        return res[::-1]
    
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        """ 199.二叉树的右视图 """
        # 时间:O(n) 空间:(n)
        from collections import deque

        res = []
        if root is None:
            return res
        queue = deque([root])
        while queue:
            n = len(queue)
            res_level = []
            for _ in range(n):
                node = queue.popleft()
                res_level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(res_level)
        return [s[-1] for s in res]
    
    def averageOfLevels(self, root: Optional[TreeNode]) -> List[float]:
        """ 637.二叉树的层平均值 """
        # 时间:O(n^2), 若使用deque则为O(n) 空间:O(n)
        res = []
        if root is None:
            return res
        queue = [root]
        while queue:
            res_level = []
            for _ in range(len(queue)):         # for开始前, Python会自动锁定len(queue)
                node = queue.pop(0)
                res_level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(sum(res_level) / len(res_level))
        return res
    
    def levelOrder(self, root: 'Node') -> List[List[int]]:      
        # 'Node'字符串形式的类型注解, 为了支持前向引用. Python中定义要在引用之前完成, 直接使用 Node 会报错, 使用 'Node' 则会延迟对Node的解析直到加载完毕
        """ 429.N叉树的层序遍历 """
        # 时间:O(n), 因为使用了deque 空间:O(n)
        from collections import deque

        res = []
        if not root:
            return res
        queue = deque([root])
        while queue:
            res_level = []
            for _ in range(len(queue)):     # Python中, for开始前锁死len(queue)
                node = queue.popleft()
                res_level.append(node.val)
                for child in node.children:
                    if child:
                        queue.append(child)
            res.append(res_level)
        return res
    
    def largestValues(self, root: Optional[TreeNode]) -> List[int]:
        """ 515.在每个树行中找最大值 """
        # 时间:O(n), 因为使用了deque 空间:O(n)
        from collections import deque

        res = []
        if not root:
            return res
        queue = deque([root])
        while queue:
            res_level = []
            for _ in range(len(queue)):     # Python中, for开始前会锁死len(queue)
                node = queue.popleft()
                res_level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(max(res_level))
        return res
    
    def connect(self, root: 'Optional[Node_r]') -> 'Optional[Node_r]':
        """ 116.填充每个节点的下一个右侧节点指针 """
        # 参考答案, 简化代码
        # 时间:O(n^2), 因为没使用deque 空间:O(n)
        if not root:
            return root
        queue = [root]
        while queue:
            pre = None          # 本层第一个节点
            for _ in range(len(queue)):
                node = queue.pop(0)
                if pre is None:
                    pre = node
                else:
                    pre.next = node
                    pre = node
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return root

        # 跌跌撞撞写对了
        # if not root:
        #     return root
        # queue = [root]
        # while queue:
        #     n = len(queue)
        #     pre = queue.pop(0)
        #     if pre.left:
        #         queue.append(pre.left)
        #     if pre.right:
        #         queue.append(pre.right)
        #     for _ in range(1, n):
        #         node = queue.pop(0)
        #         pre.next = node
        #         pre = node
        #         if node.left:
        #             queue.append(node.left)
        #         if node.right:
        #             queue.append(node.right)
        # return root

    def connect(self, root: 'Node_r') -> 'Node_r':
        """ 117.填充每个节点的下一个右侧节点指针II """
        # 时间:O(n) 空间:O(n)
        from collections import deque

        if not root:
            return root
        queue = deque([root])
        while queue:
            pre = None
            for _ in range(len(queue)):
                node = queue.popleft()
                if not pre:
                    pre = node
                else:
                    pre.next = node
                    pre = node
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return root
    
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        """ 104.二叉树的最大深度 """
        # 时间:O(n) 空间:O(n)
        # from collections import deque

        # res = []
        # if not root:
        #     return 0
        # queue = deque([root])
        # while queue:
        #     res_level = []
        #     for _ in range(len(queue)):
        #         node = queue.popleft()
        #         res_level.append(node.val)
        #         if node.left:
        #             queue.append(node.left)
        #         if node.right:
        #             queue.append(node.right)
        #     res.append(res_level)
        # return len(res)
    
        # 递归 后序遍历--求高度
        # 根节点的高度, 即为二叉树的深度, 所以递归后序遍历可以解此题
        def func(node):
            if not node:
                return 0
            return 1 + max(func(node.left), func(node.right))
        
        return func(root)

        # 递归 先序遍历--求深度
        # 涉及回溯, 以后搞

        # 再来一遍_二叉树最大深度_后序递归
        # def func(node):
        #     if node is None:
        #         return 0
        #     return 1 + max(func(node.left), func(node.right))
        # return func(root)
    
    def minDepth(self, root: Optional[TreeNode]) -> int:
        """ 111.二叉树的最小深度 """
        # 时间:O(n) 空间:O(n)
        # from collections import deque

        # res = 0
        # if not root:
        #     return 0
        # queue = deque([root])
        # while queue:
        #     res += 1
        #     for _ in range(len(queue)):
        #         node = queue.popleft()
        #         if not (node.left or node.right):
        #             return res
        #         if node.left:
        #             queue.append(node.left)
        #         if node.right:
        #             queue.append(node.right)
        # return res

        # 递归 后序遍历--求高度
        # 参考 104.二叉树的最大深度. 有点难哦
        # def func(node):
        #     if not node:
        #         return 0
        #     # return 1 + min(func(node.left), func(node.right))       # 必错无疑! 如果根节点只有一个子树呢, 这就错了
        #     if node.left and not node.right:
        #         return 1 + func(node.left)
        #     elif not node.left and node.right:
        #         return 1 + func(node.right)
        #     elif not node.left and not node.right:
        #         return 1
        #     elif node.left and  node.right:
        #         return 1 + min(func(node.left), func(node.right))
        
        # return func(root)

        # 再来一遍_二叉树的最小深度_后序递归
        def func(node):
            if not node:
                return 0
            
            if node.left and not node.right:
                return 1 + func(node.left)
            elif not node.left and node.right:
                return 1 + func(node.right)
            elif not node.left and not node.right:
                return 1
            else:
                return 1 + min(func(node.left), func(node.right))
        return func(root)
    
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """ 226.翻转二叉树 """
        # 时间:O(n) 空间:O(n)
        # from collections import deque

        # if not root:
        #     return root
        # queue = deque([root])
        # while queue:
        #     for _ in range(len(queue)):
        #         node = queue.popleft()
        #         node.left, node.right = node.right, node.left
        #         if node.left:
        #             queue.append(node.left)
        #         if node.right:
        #             queue.append(node.right)
        # return root

        # 尝试递归 前序遍历
        # 时间:O(n) 每个节点都调用一次递归 递归中的操作只是交换一下, 没有其他花哨操作 空间:最差, 链式二叉树 O(n); 最好, 完全平衡二叉树O(logn)
        # def func(node):
        #     if not node:
        #         return
        #     node.left, node.right = node.right, node.left
        #     func(node.left)
        #     func(node.right)
        
        # func(root)
        # return root

        # 尝试迭代法 前序遍历
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
        """ 589.N叉树的前序遍历 """
        # 递归 前序遍历
        # 时间:O(n) 每个节点都调用一次, 处理操作仅收集节点, 没有其他花哨操作 空间:O(n)
        def func(node, res):
            if not node:
                return
            res.append(node.val)
            for _nd in node.children:
                func(_nd, res)
        
        res = []
        func(root, res)
        return res
    
    def postorder(self, root: 'Node') -> List[int]:
        """ 590.N叉树的后序遍历 """
        # 时间:O(n) 空间:O(n)
        def func(node, res):
            if not node:
                return
            for _nd in node.children:
                func(_nd, res)
            res.append(node.val)
        
        res = []
        func(root, res)
        return res
    
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        """ 101.对称二叉树 \n
            判断对称二叉树可不是判断左右两个子节点, 而是要判断左右两棵子树是否为互相翻转的 \n
            通过判断左右子树的返回值判断二者内侧外侧节点是否相等, 所以 后序遍历 \n
            左子树--左右中, 右子树--右左中 , 总体看来是后序遍历 """
        # 时间:O(n) 空间:O(1)
        # def func(left, right):
        #     # 终止条件
        #     if left and not right:
        #         return False
        #     elif not left and right:
        #         return False
        #     elif not left and not right:
        #         return True
        #     elif left.val != right.val:
        #         return False
        #     else:
        #         return func(left.left, right.right) and func(left.right, right.left)
        
        # return func(root.left, root.right)      # 题目已说明root不空
    
        # 参考答案, 队列
        # 还是那些条件, 灵活调整而已
        # queue = [root.left, root.right]
        # while queue:
        #     left, right = queue.pop(0), queue.pop(0)

        #     if left and not right:
        #         return False
        #     elif not left and right:
        #         return False
        #     elif not left and not right:
        #         continue
        #     elif left.val != right.val:
        #         return False

        #     queue.extend([left.left, right.right, left.right, right.left])
        # return True
    
        # 参考答案, 栈
        # 其实就是上面队列 pop() 从末尾弹出

        # 再来一遍_只能后续递归
        # def func(left, right):
        #     """ 判断两棵子树是否对称, 所以传入两棵子树根节点 """
        #     if not left and right:
        #         return False
        #     elif left and not right:
        #         return False
        #     elif not left and not right:
        #         return True
        #     elif left.val != right.val:
        #         return False
        #     else:
        #         return func(left.left, right.right) and func(left.right, right.left)
        # if not root:
        #     return True
        # return func(root.left, root.right)

        # 再来一遍_迭代法_队列
        # 还是尽量保持和上面写法一致吧
        # if not root:
        #     return True
        # queue = [root.left, root.right]
        # while queue:
        #     left, right = queue.pop(0), queue.pop(0)
        #     if not (left or right):
        #         continue
            
        #     # 相同的判断条件
        #     if not left and right:
        #         return False
        #     elif left and not right:
        #         return False
        #     elif left.val != right.val:
        #         return False
            
        #     queue.extend([left.left, right.right, left.right, right.left])
        # return True

        # Again_只能后序递归_因为要通过左右子树计算当前树的结果
        # 时间:O(n) 空间:O(n)
        # def func(left, right):
        #     if left and not right:
        #         return False
        #     elif not left and right:
        #         return False
        #     elif not (left or right):
        #         return True
        #     elif left.val != right.val:
        #         return False
        #     else:
        #         return func(left.left, right.right) and func(left.right, right.left)
        
        # return func(root.left, root.right)

        # Again_迭代法_判断逻辑相同
        # 时间:O(n) 空间:O(n)
        queue = [[root.left, root.right]]
        while queue:
            left, right = queue.pop(0)
            if left and not right:
                return False
            elif not left and right:
                return False
            elif not (left or right):
                continue
            elif left.val != right.val:
                return False
            queue.append([left.left, right.right])
            queue.append([left.right, right.left])
        return True
    
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        """ 100.相同的树 \n
            即 101.对称二叉树 """
        # 递归
        def func(left, right):
            # 终止条件
            if not left and right:
                return False
            elif left and not right:
                return False
            elif not left and not right:
                return True
            elif left.val != right.val:
                return False
            else:
                return func(left.left, right.left) and func(left.right, right.right)
        
        return func(p, q)
    
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        """ 572.另一棵树的子树 \n
            即 100.相同的树 101.对称二叉树 """
        # 递归, 都快背过了
        # def func(left, right):
        #     # 终止条件
        #     if not left and right:
        #         return False
        #     elif left and not right:
        #         return False
        #     elif not left and not right:
        #         return True
        #     elif left.val != right.val:
        #         return False
        #     else:
        #         return func(left.left, right.left) and func(left.right, right.right)
            
        # if not (root or subRoot):
        #     return True
        # elif not (root and subRoot):
        #     return False
        # return func(root, subRoot) or self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)
        
        # 再来一遍_后序递归
        def func(left, right):
            if not left and right:
                return False
            elif left and not right:
                return False
            elif not left and not right:
                return True
            elif left.val != right.val:
                return False
            else:
                return func(left.left, right.left) and func(left.right, right.right)
        
        if not (root or subRoot):
            return True
        elif not (root and subRoot):
            return False
        return func(root, subRoot) or self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)
    
    def maxDepth(self, root: 'Node') -> int:
        """ 559.N叉树的最大深度 """
        # 层序遍历
        # 时间:O(n) 空间:O(n)
        # from collections import deque

        # if not root:
        #     return 0
        # res = 0             # 深度
        # queue = deque([root])
        # while queue:
        #     res += 1
        #     for _ in range(len(queue)):
        #         node = queue.popleft()
        #         for _nd in node.children:
        #             if _nd:
        #                 queue.append(_nd)
        # return res
    
        # 递归 后序遍历--求高度, 根节点高度刚好是树的最大深度
        # 参考 104.二叉树的最大深度 递归解法
        def func(node: Optional['Node']) -> Optional[int]:
            if not node:
                return 0
            return 1 + max([func(_nd) for _nd in node.children]) if node.children else 1
        
        return func(root)
    
    def countNodes(self, root: Optional[TreeNode]) -> int:
        """ 222.完全二叉树的节点个数 """
        # 时间:O(n) 空间:O(logn) 算上递归栈调用开销
        # 参考答案 递归 后序遍历, 因为要用左右节点的返回值计算以当前节点为根节点的子树的节点数量
        def func(node):
            if not node:
                return 0
            leftNum = func(node.left)
            rightNum = func(node.right)
            return 1 + leftNum + rightNum

        return func(root)

        # 不推荐, 隐藏了一些逻辑--简化版代码, 自己也是倾向写出这样的代码
        # def func(node):
        #     if not node:
        #         return 0
        #     return func(node.left) + func(node.right) + 1
        
        # return func(root)

        # 利用 完全二叉树 的性质, 没有看懂

    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        """ 110.平衡二叉树 \n
            因为求深度可以从上到下去查 所以需要前序遍历（中左右）
            而高度只能从下到上去查，所以只能后序遍历（左右中）"""
        # 从左右子树高度差来判断. 后序递归 求 高度
        # def func(node):
        #     if not node:
        #         return 0
            
        #     leftHeight = func(node.left)
        #     if leftHeight == -1:
        #         return -1
        #     rightHeight = func(node.right)
        #     if rightHeight == -1:
        #         return -1
            
        #     if abs(leftHeight - rightHeight) > 1:
        #         return -1
        #     return 1 + max(leftHeight, rightHeight)
        # return False if func(root) == -1 else True

        # 再来一遍
        # def func(node):
        #     if not node:
        #         return 0
            
        #     leftHeight = func(node.left)
        #     if leftHeight == -1:
        #         return -1
        #     rightHeight = func(node.right)
        #     if rightHeight == -1:
        #         return -1
            
        #     if abs(leftHeight - rightHeight) > 1:
        #         return -1
        #     return 1 + max(leftHeight, rightHeight)

        # return False if func(root) == -1 else True
    
        # 再来一遍_二叉平衡树_后序递归
        # 果然, 你没记住(狗头)
        def func(node):
            if node is None:
                return 0
            
            leftHeight = func(node.left)
            if leftHeight == -1:
                return -1
            rightHeight = func(node.right)
            if rightHeight == -1:
                return -1
            if abs(leftHeight - rightHeight) > 1:
                return -1
            return 1 + max(leftHeight, rightHeight)
        return False if func(root) == -1 else True
    
    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
        """ 257.二叉树的所有路径 \n
            父节点指向子节点的路径--先序递归/回溯 """
        # def func(node, path, res):
        #     path.append(node.val)       # 就是这里有点特殊, 写在了终止条件之前, 想想确是如此
        #     # 终止条件
        #     if not (node.left or node.right):
        #         res.append("->".join([str(val) for val in path]))
        #         return
        #     # 单层递归逻辑
        #     if node.left:
        #         func(node.left, path, res)
        #         path.pop()
        #     if node.right:
        #         func(node.right, path, res)
        #         path.pop()
        
        # path = []
        # res = []
        # if not root:
        #     return res
        # func(root, path, res)
        # return res

        # 尝试迭代法_参考答案
        # 路径一般先序遍历, 方便记录父节点指向子节点
        if not root:
            return []
        stack = [root]
        path_st = [[root.val]]
        res = []
        while stack:
            node = stack.pop()
            path = path_st.pop()
            if not (node.left or node.right):
                res.append('->'.join([str(p) for p in path]))
            if node.right:
                stack.append(node.right)
                path_st.append(path + [node.right.val])
            if node.left:
                stack.append(node.left)
                path_st.append(path + [node.left.val])
        return res
    
    def sumOfLeftLeaves(self, root: Optional[TreeNode]) -> int:
        """ 404.左叶子之和 \n
            当前节点必须是父节点, 才能判断是否有左叶子节点"""
        # 迭代法_更容易理解
        # 二叉树前中后迭代均可, 也是以当前节点作父节点, 判断左节点是否为左叶子
        # if not root:
        #     return 0
        # res = []
        # stack = [root]
        # while stack:
        #     node = stack.pop()
        #     if node.left and not (node.left.left or node.left.right):
        #         res.append(node.left.val)
        #     if node.right:
        #         stack.append(node.right)
        #     if node.left:
        #         stack.append(node.left)
        # return sum(res)

        # 后序递归_因为要用左右子树的结果计算当前节点左叶子之和
        # 终止条件
        # if not root:
        #     return 0
        # elif not root.left and not root.right:
        #     return 0
        # # 单层递归逻辑
        # leftVal = self.sumOfLeftLeaves(root.left)
        # if root.left and root.left.left is None and root.left.right is None:
        #     leftVal = root.left.val
        # rightVal = self.sumOfLeftLeaves(root.right)
        # return leftVal + rightVal
    
        # 再来一遍_迭代法_左叶子之和
        # if not root:
        #     return 0
        # res = []
        # stack = [root]
        # while stack:
        #     node = stack.pop(0)
        #     if node.left and not (node.left.left or node.left.right):
        #         res.append(node.left.val)
        #     if node.right:
        #         stack.append(node.right)
        #     if node.left:
        #         stack.append(node.left)
        # return sum(res)

        # 再来一遍_后序递归_要通过左右节点的结果来计算当前节点的结果, 牛逼, 竟然写出来了
        def func(node):
            if not node:
                return 0
            elif not (node.left or node.right):
                return 0
            leftRes = func(node.left)
            if node.left and not (node.left.left or node.left.right):
                leftRes = node.left.val
            rightRes = func(node.right)
            return leftRes + rightRes
        
        return func(root)
    
    def findBottomLeftValue(self, root: Optional[TreeNode]) -> int:
        """ 513.找树左下角的值 """
        # 时间:O(n) 空间:O(n)
        # from collections import deque

        # res = []
        # queue = deque([root])
        # while queue:
        #     res_level = []
        #     for _ in range(len(queue)):     # Python特性, for开始前锁定len(queue)长度
        #         node = queue.popleft()
        #         res_level.append(node.val)
        #         if node.left:
        #             queue.append(node.left)
        #         if node.right:
        #             queue.append(node.right)
        #     res.append(res_level)
        # return res[-1][0]

        # 再来一遍
        from collections import deque

        queue = deque([root])
        res = []
        while queue:
            res_level = []
            for _ in range(len(queue)):
                node = queue.popleft()
                res_level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(res_level)
        return res[-1][0]
    
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        """ 112.路径总和 """
        # 递归函数什么时候需要返回值?
        #   如果需要搜索搜索整棵二叉树且不用处理递归返回值, 递归函数就不要返回值
        #   如果需要搜索整棵二叉树且需要处理递归返回值, 递归函数就要返回值
        #   如果要搜索一条符合条件的路径, 那一定需要返回值, 因为找到了符合条件的路径就要及时返回 --> 本题
        
        # 时间:O(n), 每个节点可能都会遍历到 空间:O(n) 递归栈最大深度为树的高度, 二叉平衡树则为logn, 完全退化的树则为n
        # def func(node, count):
        #     # 本题不需要遍历所有节点, 所以返回值为bool
        #     # 终止条件
        #     if not (node.left or node.right) and count == 0:        # 遍历到叶子节点且路径和为targetSum
        #         return True
        #     elif not (node.left or node.right):                     # 遍历到叶子节点且路径和不为targetSum
        #         return False
        #     # 单层递归
        #     if node.left:
        #         if func(node.left, count - node.left.val):          # 体现回溯
        #             return True
        #     if node.right:
        #         if func(node.right, count - node.right.val):        # 体现回溯
        #             return True
        #     return False
        
        # if not root:
        #     return False
        # return func(root, targetSum - root.val)
        
        # 迭代_前序遍历_用栈实现递归
        # 时间:O(n) 空间:O(n)
        # if not root:
        #     return False
        # stack = [[root, root.val]]
        # while stack:
        #     node, val = stack.pop()
        #     if not (node.left or node.right) and val == targetSum:
        #         return True
        #     if node.right:
        #         stack.append([node.right, val + node.right.val])
        #     if node.left:
        #         stack.append([node.left, val + node.left.val])
        # return False

        # 再来一遍_路径总和
        # 递归函数什么时候需要返回值:
        #   如果需要搜索整棵树, 通常不需要返回值
        #   如果需要搜索一条符合条件的路径, 通常需要返回值, 因为找到要及时返回. 返回bool类型就是为了, 找到就立刻返回
        # def func(node, count):
        #     # 重新来做时, 自己没能理清终止条件--遍历到叶子节点判断是否找到
        #     # 终止条件的返回值用来控制递归终止, 不是为返回题目结果
        #     if count == 0 and not (node.left or node.right):
        #         return True
        #     if count != 0 and not (node.left or node.right):
        #         return False
            
        #     # 递归逻辑的最后一层返回值才是题目结果
        #     if node.left:
        #         if func(node.left, count - node.left.val):
        #             return True
        #     if node.right:
        #         if func(node.right, count - node.right.val):
        #             return True
        #     return False
        
        # if not root:
        #     return False
        # return func(root, targetSum - root.val)

        # 自己尝试迭代_路径总和
        if not root:
            return False
        stack = [[root, root.val]]
        while stack:
            node, _sum = stack.pop()
            if _sum == targetSum and not (node.left or node.right):
                return True
            if node.right:
                stack.append([node.right, _sum + node.right.val])
            if node.left:
                stack.append([node.left, _sum + node.left.val])
        return False
    
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        """ 113.路径总和II \n
            与 112.路径总和 区别: 113.要找到所有路径, 要遍历整棵树, 所以不要返回值! """
        # 时间:O(n), 遍历所有节点 空间:O(n), 递归栈为最大
        # def func(node, count, path, res):
        #     if not (node.left or node.right) and count == 0:        # 刚好找到路径
        #         res.append(path[:])
        #         return
        #     elif not (node.left or node.right):
        #         return
            
        #     if node.left:
        #         func(node.left, count - node.left.val, path + [node.left.val], res)
        #     if node.right:
        #         func(node.right, count - node.right.val, path + [node.right.val], res)
        
        # if not root:
        #     return []
        # path = [root.val]
        # res = []
        # func(root, targetSum - root.val, path, res)
        # return res

        # Again_前序递归_注意回溯
        # def func(node, count, path, res):
        #     path.append(node.val)       # 要在终止条件之前加, 每次添加完都要通过终止条件判断是否到达叶子节点
        #     if count == 0 and not (node.left or node.right):
        #         res.append(path[:])
        #         return
        #     if count != 0 and not (node.left or node.right):
        #         return
        #     if node.left:
        #         func(node.left, count - node.left.val, path, res)
        #         path.pop()
        #     if node.right:
        #         func(node.right, count - node.right.val, path, res)
        #         path.pop()
        
        # if not root:
        #     return []
        # res = []
        # path = []
        # func(root, targetSum - root.val, path, res)
        # return res

        # Again_前序递归_隐藏回溯
        def func(node, count, path, res):
            # 终止条件_上来就终止条件, 所以在调用递归时修改path, 隐藏了回溯逻辑
            if count == 0 and not (node.left or node.right):
                res.append(path[:])
                return
            elif count != 0 and not (node.left or node.right):
                return
            
            if node.left:
                func(node.left, count - node.left.val, path + [node.left.val], res)
            if node.right:
                func(node.right, count - node.right.val, path + [node.right.val], res)
        
        if not root:
            return []
        path = []
        res = []
        func(root, targetSum - root.val, [root.val], res)
        return res

    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        """ 106.从中序与后序遍历序列构造二叉树 \n
            切割 """
        # 时间:O(n) 空间:O(n)
        # if not postorder:
        #     return None
        
        # rootVal = postorder[-1]
        # root = TreeNode(rootVal)
        # if len(postorder) == 1:
        #     return root
        
        # ind_inorder = inorder.index(rootVal)
        # left_inorder = inorder[:ind_inorder]
        # right_inorder = inorder[ind_inorder + 1:]

        # left_postorder = postorder[:len(left_inorder)]
        # right_postorder = postorder[len(left_inorder):-1]

        # root.left = self.buildTree(left_inorder, left_postorder)
        # root.right = self.buildTree(right_inorder, right_postorder)

        # return root

        # Again
        # 时间:O(n) 空间:O(n)
        if len(postorder) == 0:
            return None
        rootVal = postorder[-1]
        root = TreeNode(rootVal)
        if len(postorder) == 1:
            return root
        
        ind_inorder = inorder.index(rootVal)
        left_inorder = inorder[:ind_inorder]
        right_inorder = inorder[ind_inorder + 1:]

        left_postorder = postorder[:len(left_inorder)]
        right_postorder = postorder[len(left_postorder):-1]

        root.left = self.buildTree(left_inorder, left_postorder)
        root.right = self.buildTree(right_inorder, right_postorder)
        return root
    
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        """ 105.从前序与中序遍历序列构造二叉树 """
        # 时间:O(n^2), 因为inorder.index()的时间为O(n) 空间:O(n)
        # if len(preorder) == 0:
        #     return None
        
        # rootVal = preorder[0]
        # root = TreeNode(rootVal)
        # if len(preorder) == 1:
        #     return root
        
        # ind_inorder = inorder.index(rootVal)
        # left_inorder = inorder[:ind_inorder]
        # right_inorder = inorder[ind_inorder + 1:]

        # left_preorder = preorder[1:1 + len(left_inorder)]
        # right_preorder = preorder[1 + len(left_inorder):]

        # root.left = self.buildTree(left_preorder, left_inorder)
        # root.right = self.buildTree(right_preorder, right_inorder)

        # return root

        # Again_
        # 时间:O(n^2) 空间:O(n)
        if len(preorder) == 0:
            return None
        rootVal = preorder[0]
        root = TreeNode(rootVal)
        if len(preorder) == 1:
            return root

        ind_inorder = inorder.index(rootVal)
        left_inorder = inorder[:ind_inorder]
        right_inorder = inorder[ind_inorder + 1:]

        left_preorder = preorder[1:1 + len(left_inorder)]
        right_preorder = preorder[1 + len(left_inorder):]

        root.left = self.buildTree(left_preorder, left_inorder)
        root.right = self.buildTree(right_preorder, right_inorder)
        return root

    def constructMaximumBinaryTree(self, nums: List[int]) -> Optional[TreeNode]:
        """ 654.最大二叉树 """
        # if len(nums) == 1:
        #     return TreeNode(nums[0])
        # maxVal = max(nums)
        # maxInd = nums.index(maxVal)
        # root = TreeNode(maxVal)
        # if maxInd > 0:
        #     root.left = self.constructMaximumBinaryTree(nums[:maxInd])
        # if maxInd < len(nums) - 1:
        #     root.right = self.constructMaximumBinaryTree(nums[maxInd + 1:])
        # return root

        # Again_前序递归
        # 时间:O(n^2) 空间:O(n)
        if len(nums) == 1:
            return TreeNode(nums[0])
        
        ind_max = nums.index(max(nums))
        root = TreeNode(nums[ind_max])
        if ind_max > 0:
            root.left = self.constructMaximumBinaryTree(nums[:ind_max])
        if ind_max < len(nums) - 1:
            root.right = self.constructMaximumBinaryTree(nums[ind_max + 1:])
        return root
    
    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
        """ 617.合并二叉树 """
        # 同时遍历两棵树而已(虽然自己没写出来)_先序递归[推荐]
        # 时间:O(n) 空间:O(n)
        # if not root1:
        #     return root2
        # elif not root2:
        #     return root1
        
        # root1.val += root2.val
        # root1.left = self.mergeTrees(root1.left, root2.left)
        # root1.right = self.mergeTrees(root1.right, root2.right)
        # return root1
    
        # 中序递归
        # 时间:O(n) 空间:O(n)
        # if not root1:
        #     return root2
        # elif not root2:
        #     return root1
        
        # root1.left = self.mergeTrees(root1.left, root2.left)
        # root1.val += root2.val
        # root1.right = self.mergeTrees(root1.right, root2.right)
        # return root1

        # 后序递归, 也是可以滴. 略

        # 迭代法_参考 101.对称二叉树, 一般同时操作两棵树, 都是使用队列层序遍历
        # 比较左右两棵子树, 而不是左右节点
        # 时间:O(n) 空间:O(n)
        if not root1:
            return root2
        elif not root2:
            return root1
        
        queue = [[root1, root2]]
        while queue:
            left, right = queue.pop(0)
            left.val += right.val

            if left.left and right.left:
                queue.append([left.left, right.left])
            if left.right and right.right:
                queue.append([left.right, right.right])
            if not left.left and right.left:
                left.left = right.left
            if not left.right and right.right:
                left.right = right.right
        return root1
    
    def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        """ 700.二叉搜索树中的搜索 """
        # 自己写对了, 感觉条件略有复杂, 可以看下答案
        # 时间:O(logn), 若普通二叉树则O(n), 退化为链表 空间:O(logn) 若普通二叉树则O(n)
        # if root.val == val:
        #     return root
        # elif not (root.left or root.right):
        #     return None
        
        # if root.val > val and root.left:
        #     return self.searchBST(root.left, val)
        # elif root.val < val and root.right:
        #     return self.searchBST(root.right, val)

        # 迭代法
        # 时间:O(logn) 空间:O(1)
        if not root:
            return None
        cur = root
        while cur:
            if cur.val == val:
                return cur
            elif cur.val > val:
                cur = cur.left
            elif cur.val < val:
                cur = cur.right
        return cur
    
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        """ 98.验证二叉搜索树 \n
            二叉搜索树的中序遍历是递增的 """
        # 中序递归_牛逼
        # def func(node, res):
        #     if node.left:
        #         func(node.left, res)
        #     res.append(node.val)
        #     if node.right:
        #         func(node.right, res)
        
        # res = []
        # func(root, res)
        # for i in range(1, len(res)):
        #     if res[i - 1] >= res[i]:
        #         return False
        # return True

        # 中序迭代_要复习
        # 时间:O(n) 空间:O(n)
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
        for i in range(1, len(res)):
            if res[i - 1] >= res[i]:
                return False
        return True
    
    def getMinimumDifference(self, root: Optional[TreeNode]) -> int:
        """ 530.二叉搜索树的最小绝对差 \n
            二叉搜索树中序遍历严格递增 """
        # 迭代_中序遍历
        # 时间:O(n) 空间:O(n)
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
        # min_diff = float('inf')
        # for i in range(1, len(res)):
        #     min_diff = min(min_diff, res[i] - res[i - 1])
        # return min_diff

        # 递归_中序遍历
        # 时间:O(n) 空间:O(n)
        # res = []
        # def func(node):
        #     if not node:
        #         return
        #     func(node.left)
        #     res.append(node.val)
        #     func(node.right)
        
        # func(root)
        # min_diff = float('inf')
        # for i in range(1, len(res)):
        #     min_diff = min(min_diff, res[i] - res[i - 1])
        # return min_diff

        # 递归_中序遍历_记录前一个指针
        pre = None
        min_diff = float('inf')
        def func(node):
            nonlocal pre, min_diff      # 声明使用外部作用域的变量
            if not node:
                return
            func(node.left)
            if pre:
                min_diff = min(min_diff, node.val - pre.val)
            pre = node
            func(node.right)
        
        func(root)
        return min_diff
    
    def findMode(self, root: Optional[TreeNode]) -> List[int]:
        """ 501.二叉搜索数中的众数 """
        # 基于 中序递归
        # 时间:O(n) 空间:O(n)
        # res = []
        # max_count = 0
        # count = 0
        # pre = None
        # def func(node):
        #     nonlocal count, pre, max_count, res

        #     if not node:
        #         return
        #     func(node.left)
        #     if not pre:
        #         count = 1
        #     elif pre.val == node.val:
        #         count += 1
        #     elif pre.val != node.val:
        #         count = 1
        #     pre = node
        #     if count == max_count:
        #         res.append(node.val)
        #     elif count > max_count:
        #         max_count = count
        #         res = [node.val]
        #     func(node.right)
        
        # func(root)
        # return res

        # 基于_中序迭代
        # 时间:O(n) 空间:O(n)
        pre = None
        count = 0
        max_count = 0
        res = []

        stack = []
        cur = root
        while cur or stack:
            if cur:
                stack.append(cur)
                cur = cur.left
            else:
                cur = stack.pop()
                # 处理逻辑与上面相同
                if not pre:
                    count = 1
                elif pre.val == cur.val:
                    count += 1
                elif pre.val != cur.val:
                    count = 1
                pre = cur
                if count == max_count:
                    res.append(cur.val)
                elif count > max_count:
                    max_count = count
                    res = [cur.val]
                cur = cur.right
        return res
    
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        """ 236.二叉树的最近公共祖先 """
        # 终止条件
        # 时间:O(n) 空间:O(n)
        if not root:
            return root
        elif root in [p, q]:
            return root
        # 单层递归逻辑
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if left and right:
            return root
        elif not left and right:
            return right
        elif left and not right:
            return left
        
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        """ 235.二叉搜索树的最近公共祖先 \n
            二叉搜索树, 直接找就行 """
        # 迭代法
        # 时间:O(n) 空间:O(1)
        while root:
            if root.val > p.val and root.val > q.val:
                root = root.left
            elif root.val < p.val and root.val < q.val:
                root = root.right
            else:
                return root
        return None

        # 时间:O(n) 空间:O(h)
        # 终止条件
        # if not root:
        #     return root
        # # 单层递归
        # if root.val > p.val and root.val > q.val:
        #     return self.lowestCommonAncestor(root.left, p, q)       # 典型找到一条边就返回. 注意, 不能 == 
        # elif root.val < p.val and root.val < q.val:
        #     return self.lowestCommonAncestor(root.right, p, q)
        # else:
        #     return root

        # 自己实现的_但不是答案推荐
        # 时间:O(n) 空间:O(n)
        # if not root:
        #     return root
        # elif p.val <= root.val <= q.val or q.val <= root.val <= p.val:
        #     return root
        # leftRes = self.lowestCommonAncestor(root.left, p, q)
        # rightRes = self.lowestCommonAncestor(root.right, p, q)
        # if leftRes and rightRes:
        #     return root
        # elif not leftRes:
        #     return rightRes
        # elif not rightRes:
        #     return leftRes

    def insertIntoBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        """ 701.二叉搜索树中的插入操作 \n
            找val, 直到遇到空节点, 新建插入就行了 """
        # 时间:O(n) 空间:O(1)
        # if not root:
        #     return TreeNode(val)
        # cur = root
        # while cur:
        #     pre = cur
        #     if cur.val > val:
        #         cur = cur.left
        #         if not cur:
        #             pre.left = TreeNode(val)
        #             return root
        #     elif cur.val < val:
        #         cur = cur.right
        #         if not cur:
        #             pre.right = TreeNode(val)
        #             return root

        # 递归
        # 时间:O(h) 空间:O(h)
        # 终止条件
        if not root:
            return TreeNode(val)
        # 单层递归
        if root.val > val:
            root.left = self.insertIntoBST(root.left, val)
        if root.val < val:
            root.right = self.insertIntoBST(root.right, val)
        return root
    
    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        """ 450.删除二叉搜索树中的节点 """
        # 比较难, 建议先看答案
        # 动归五部曲: 1.递归函数参数和返回值, 参数:根节点, key 返回值:删除后的 根节点, 返回给上层承接
        #           2.遇到空, 没找到, 返回空
        #           3.单层递归
        # 时间:O(logn) 空间:O(logn)
        # 终止条件
        if not root:            # 没找到删除节点, 返回空
            return None
        # ---- 单层递归 ----
        if root.val == key:     # 找到删除节点
            if not (root.left or root.right):       # 删除节点为叶节点
                return None
            elif not root.left:                     # 删除节点只有 左孩子/右孩子
                return root.right
            elif not root.right:
                return root.left
            else:                                   # 删除节点有左右孩子
                cur = root.right
                while cur.left:
                    cur = cur.left
                cur.left = root.left
                return root.right
        if key < root.val:
            root.left = self.deleteNode(root.left, key)
        if key > root.val:
            root.right = self.deleteNode(root.right, key)
        # ------------------
        return root
    
    def trimBST(self, root: Optional[TreeNode], low: int, high: int) -> Optional[TreeNode]:
        """ 669.修剪二叉搜索树 \n
            是不是想直接套用 450.删除二叉搜索树中的节点, 可不是这么简单, 先看答案吧 """
        # 迭代法, 修剪分为3步: (也不是太好理解)
        #               1.将root移动到合法范围 
        #               2.修剪左子树 
        #               3.修剪右子树
        # if not root:
        #     return None
        # while root:
        #     if root.val < low:
        #         root = root.right
        #     elif root.val > high:
        #         root = root.left
        #     else:
        #         break
        
        # cur = root
        # while cur:
        #     while cur.left and cur.left.val < low:
        #         cur.left = cur.left.right
        #     cur = cur.left
        
        # cur = root
        # while cur:
        #     while cur.right and cur.right.val > high:
        #         cur.right = cur.right.left
        #     cur = cur.right
        # return root

        # 先序递归_不太好理解
        # 时间:O(logn) 空间:O(logn)
        # 终止条件
        # if not root:
        #     return None
        # # 单层递归
        # if root.val < low:
        #     right = self.trimBST(root.right, low, high)
        #     return right
        # elif root.val > high:
        #     left = self.trimBST(root.left, low, high)
        #     return left
        # root.left = self.trimBST(root.left, low, high)
        # root.right = self.trimBST(root.right, low, high)
        # return root

        # 重写一遍递归_先序递归
        # 终止条件
        if not root:
            return None
        # 单层递归
        # 若当前节点值非法
        if root.val < low:      # 根据二叉搜索树的性质, root.val < low则要舍弃左子树, 去右子树继续寻找合法节点
            right = self.trimBST(root.right, low, high)
            return right
        elif root.val > high:
            left = self.trimBST(root.left, low, high)
            return left
        # 当前节点值合法, 则修剪左右子树
        root.left = self.trimBST(root.left, low, high)
        root.right = self.trimBST(root.right, low, high)
        return root
    
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        """ 108.将有序数组转化为二叉搜索树 """
        # 时间:O(n) 空间:O(logn)
        def func(nums, left, right):
            if left > right:
                return
            mid = (left + right) // 2
            root = TreeNode(nums[mid])
            root.left = func(nums, left, mid - 1)
            root.right = func(nums, mid + 1, right)
            return root
        return func(nums, 0, len(nums) - 1)
    
    def convertBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """ 538.把二叉搜索树转化为累加树 \n
            二叉搜索树是有序的, 累加树即右中左访问得到数组后, 从后往前累加 """
        # 反中序递归
        # 时间:O(n) 空间:O(logn)
        # pre = 0     # 前一个节点值

        # def func(cur):
        #     if not cur:
        #         return
        #     func(cur.right)
        #     nonlocal pre
        #     cur.val += pre
        #     pre = cur.val
        #     func(cur.left)
        
        # func(root)
        # return root

        # 反中序迭代_忘了吧, 还是要复习
        # 时间:O(n) 空间:O(n)
        if not root:
            return None
        stack = []
        cur = root
        pre = 0         # 前一个节点值
        while cur or stack:
            if cur:
                stack.append(cur)
                cur = cur.right
            else:
                cur = stack.pop()
                cur.val += pre
                pre = cur.val
                cur = cur.left
        return root