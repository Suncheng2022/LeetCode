"""
07.25   是的, 该走了. 工作还是要尽力完成.
        师姐给我买了电影<狂飙>
"""
from typing import List, Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# class Node:
#     def __init__(self, val: Optional[int]=None, children: Optional[List['Node']]=None):
#         self.val = val
#         self.children = children

class Node:
    def __init__(self, val: Optional[int]=0, left: Optional['Node']=None, right: Optional['Node']=None, next: Optional['Node']=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next

class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        """ 144.二叉树的前序遍历 """
        # 时间:O(n) 因为每个节点都恰好访问一次, 操作为res.append(node.val) 复杂度O(1)
        # 空间:O(logn) 一般取决于树的高度
        # def backtrace(node, res):
        #     # 终止条件
        #     if not node:
        #         return
        #     # 递归调用逻辑
        #     res.append(node.val)
        #     backtrace(node.left, res)
        #     backtrace(node.right, res)
        
        # res = []
        # backtrace(root, res)
        # return res

        ## 迭代遍历
        # 时间:O(n) 每个节点的处理操作:出栈 取值 检查左右孩子
        # 空间:O(logn) 模拟的就是递归, 栈的最大深度也是树的深度
        if not root:
            return []
        stack = [root]
        res = []
        while stack:
            node = stack.pop(-1)
            res.append(node.val)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return res

    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        """ 94.二叉树的中序遍历 """
        # 时间:O(n) 空间:(logn)
        # res = []

        # def backtrack(node):
        #     # 终止条件
        #     if not node:
        #         return
        #     # 递归调用逻辑
        #     backtrack(node.left)
        #     res.append(node.val)
        #     backtrack(node.right)
        
        # backtrack(root)
        # return res

        ## 迭代中序
        # 时间:O(n) 每个节点压入栈/弹出栈, 处理
        # 空间:O(logn)
        res = []
        stack = []
        cur = root
        while cur or stack:
            if cur:
                stack.append(cur)
                cur = cur.left
            else:
                cur = stack.pop(-1)
                res.append(cur.val)
                cur = cur.right
        return res
    
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        """ 145.二叉树的后序遍历 """
        # 时间:O(n) 空间:O(logn)
        # def backtrack(node, res):
        #     # 终止条件
        #     if not node:
        #         return
        #     # 递归调用逻辑
        #     backtrack(node.left, res)
        #     backtrack(node.right, res)
        #     res.append(node.val)
        
        # res = []
        # backtrack(root, res)
        # return res

        ## 迭代后序
        ## 先序遍历: 中左右 --> 中 右左 --> 左右 中, 后序遍历
        if not root:
            return []
        stack = [root]
        res = []
        while stack:
            node = stack.pop(-1)
            res.append(node.val)
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
        return res[::-1]
    
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        """ 102.二叉树的层序遍历 """
        if not root:
            return []
        queue = [root]
        res = []
        while queue:
            num = len(queue)
            _level = []
            for _ in range(num):
                node = queue.pop(0)
                _level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(_level)
        return res
    
    def levelOrderBottom(self, root: Optional[TreeNode]) -> List[List[int]]:
        """ 107.二叉树的层序遍历II """
        if not root:
            return []
        queue = [root]
        res = []
        while queue:
            _num = len(queue)
            _level = []
            for _ in range(_num):
                node = queue.pop(0)
                _level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(_level)
        return res[::-1]
    
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        """ 199.二叉树的右视图 """
        if not root:
            return []
        res = []
        queue = [root]
        while queue:
            num = len(queue)
            level = []
            for _ in range(num):
                node = queue.pop(0)
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(level[-1])
        return res
    
    def averageOfLevels(self, root: Optional[TreeNode]) -> List[float]:
        """ 637.二叉树的层平均值 """
        from collections import deque

        if not root:
            return []
        queue = deque([root])
        res = []
        while queue:
            num = len(queue)
            level = []
            for _ in range(num):
                node = queue.popleft()
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(sum(level) / len(level))
        return res
    
    def levelOrder(self, root: 'Node') -> List[List[int]]:
        """ 429.N叉树的遍历 """
        from collections import deque

        if not root:
            return []
        queue = deque([root])
        res = []
        while queue:
            num = len(queue)
            level = []
            for _ in range(num):
                node = queue.popleft()
                level.append(node.val)
                for _nd in node.children:
                    if _nd:
                        queue.append(_nd)
            res.append(level)
        return res

    def largestValues(self, root: Optional[TreeNode]) -> List[int]:
        """ 515.在每个树行中找最大值 """
        from collections import deque

        if not root:
            return []
        res = []
        queue = deque([root])
        while queue:
            level_size = len(queue)
            level = []
            for _ in range(level_size):
                node = queue.popleft()
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(max(level))
        return res

    def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':      # 引号, 延迟解析
        """ 116.填充每个节点的下一个右侧节点指针 """
        from collections import deque

        if not root:
            return
        queue = deque([root])
        while queue:
            level_size = len(queue)
            level = []
            for _ in range(level_size):
                node = queue.popleft()
                level.append(node)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            for i in range(1, level_size):
                level[i - 1].next = level[i]
        return root
    
    def connect(self, root: 'Node') -> 'Node':
        """ 117.填充每个节点的下一个右侧节点指针II """
        from collections import deque

        if not root:
            return
        queue = deque([root])
        while queue:
            num = len(queue)
            level = []
            for _ in range(num):
                node = queue.popleft()
                if level:
                    level[-1].next = node
                level.append(node)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return root

    def maxDepth(self, root: Optional[TreeNode]) -> int:
        """ 104.二叉树的最大深度 """
        from collections import deque

        if not root:
            return 0
        queue = deque([root])
        res = []
        while queue:
            level_size = len(queue)
            level = []
            for _ in range(level_size):
                node = queue.popleft()
                level.append(node)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(level)
        return len(res)
    
    def minDepth(self, root: Optional[TreeNode]) -> int:
        """ 111.二叉树的最小深度 """
        from collections import deque

        if not root:
            return 0
        queue = deque([root])
        res = []
        while queue:
            level_size = len(queue)
            level = []
            for _ in range(level_size):
                node = queue.popleft()
                if not (node.left or node.right):
                    return len(res) + 1
                level.append(node)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(level)
        return len(res)

if __name__ == '__main__':
    """
    递归三要素:
        1.确定递归函数参数和返回值
        2.终止条件
        3.递归调用逻辑
    """