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
        def func(node):
            if not node:
                return 0
            # return 1 + min(func(node.left), func(node.right))       # 必错无疑! 如果根节点只有一个子树呢, 这就错了
            if node.left and not node.right:
                return 1 + func(node.left)
            elif not node.left and node.right:
                return 1 + func(node.right)
            elif not node.left and not node.right:
                return 1
            elif node.left and  node.right:
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
        queue = [root.left, root.right]
        while queue:
            left, right = queue.pop(0), queue.pop(0)

            if left and not right:
                return False
            elif not left and right:
                return False
            elif not left and not right:
                continue
            elif left.val != right.val:
                return False

            queue.extend([left.left, right.right, left.right, right.left])
        return True
    
        # 参考答案, 栈
        # 其实就是上面队列 pop() 从末尾弹出
    
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
        def func(node):
            if not node:
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