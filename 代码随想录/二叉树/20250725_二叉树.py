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

class Node:
    def __init__(self, val: Optional[int]=None, children: Optional[List['Node']]=None):
        self.val = val
        self.children = children

# class Node:
#     def __init__(self, val: Optional[int]=0, left: Optional['Node']=None, right: Optional['Node']=None, next: Optional['Node']=None):
#         self.val = val
#         self.left = left
#         self.right = right
#         self.next = next

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
        ## 层序遍历模板
        # from collections import deque

        # if not root:
        #     return 0
        # queue = deque([root])
        # res = []
        # while queue:
        #     level_size = len(queue)
        #     level = []
        #     for _ in range(level_size):
        #         node = queue.popleft()
        #         level.append(node)
        #         if node.left:
        #             queue.append(node.left)
        #         if node.right:
        #             queue.append(node.right)
        #     res.append(level)
        # return len(res)

        ## 进阶方法: 
        # 节点的深度, 根节点   到 当前节点 的最长简单路径边的数量   -- 前序递归, 求的是根节点深度
        # 节点的高度, 当前节点 到  叶节点  的最长简单路径边的数量   -- 后序递归, 求的是根节点高度
        # 所以, 根节点的高度就是二叉树的最大深度

        ## 后序递归, 求 根节点高度
        # # 终止条件
        # if not root:
        #     return 0
        # # 递归逻辑
        # left_height = self.maxDepth(root.left)
        # right_height = self.maxDepth(root.right)
        # return 1 + max(left_height, right_height)

        ## 前序递归, 求 根节点深度
        max_depth = 0
        def backtrack(node, depth):
            nonlocal max_depth
            max_depth = max(max_depth, depth)
            if not node:
                return
            if node.left:
                depth += 1
                backtrack(node.left, depth)
                depth -= 1
            if node.right:
                depth += 1
                backtrack(node.right, depth)
                depth -= 1

        if not root:
            return 0
        backtrack(root, 1)
        return max_depth
    
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
    
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """ 226.翻转二叉树 """
        # def backtracking(node):
        #     if not node:
        #         return
        #     node.left, node.right = node.right, node.left
        #     backtracking(node.left)
        #     backtracking(node.right)
        
        # backtracking(root)
        # return root

        ## 层序遍历也是可以的
        from collections import deque

        if not root:
            return root
        queue = deque([root])
        while queue:
            level_size = len(queue)
            for _ in range(level_size):
                node = queue.popleft()
                node.left, node.right = node.right, node.left
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return root
    
    def preorder(self, root: 'Node') -> List[int]:
        """ 589.N叉树的前序遍历 """
        res = []
        def backtrack(node):
            # 终止条件
            if not node:
                return
            # 递归调用逻辑
            res.append(node.val)
            for child in node.children:
                backtrack(child)
        
        backtrack(root)
        return res
    
    def postorder(self, root: 'Node') -> List[int]:
        """ 590.N叉树的后序遍历 """
        res = []
        def backtrack(node):
            if not node:
                return
            for child in node.children:
                backtrack(child)
            res.append(node.val)
        
        backtrack(root)
        return res
    
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        """ 101.对称二叉树 """
        ## 递归 只能后序遍历, 要通过左右节点返回值判断是否是对称二叉树
        # def backtrack(left, right):
        #     # 终止条件
        #     if not left and right:
        #         return False
        #     elif left and not right:
        #         return False
        #     elif not (left and right):
        #         return True
        #     elif left.val != right.val:
        #         return False
        #     else:
        #         return backtrack(left.left, right.right) and backtrack(left.right, right.left)
        
        # return backtrack(root.left, root.right)

        ## 迭代方法 可不是层序遍历
        # queue = [[root.left, root.right]]
        # while queue:
        #     left, right = queue.pop(0)
        #     if not left and right:
        #         return False
        #     elif left and not right:
        #         return False
        #     elif not (left and right):
        #         continue
        #     elif left.val != right.val:
        #         return False
        #     else:
        #         queue.append([left.left, right.right])
        #         queue.append([left.right, right.left])
        # return True

        ## Again 递归
        # def backtrack(left, right):
        #     if not left and right:
        #         return False
        #     elif left and not right:
        #         return False
        #     elif not (left or right):
        #         return True
        #     elif left.val != right.val:
        #         return False
        #     else:
        #         return backtrack(left.left, right.right) and backtrack(left.right, right.left)
        
        # return backtrack(root.left, root.right)

        ## Again 迭代 可不是层序遍历
        queue = [[root.left, root.right]]
        while queue:
            left, right = queue.pop(0)
            if not left and right:
                return False
            elif left and not right:
                return False
            elif not (left or right):
                continue
            elif left.val != right.val:
                return False
            else:
                queue.append([left.left, right.right])
                queue.append([left.right, right.left])
        return True
    
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        """ 100.相同的树 """
        ## 递归 后序遍历, 需要返回值判断
        # if not p and q:
        #     return False
        # elif p and not q:
        #     return False
        # elif not (p or q):
        #     return True
        # elif p.val != q.val:
        #     return False
        # else:
        #     return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)

        ## 迭代 可不是层序遍历
        queue = [[p, q]]
        while queue:
            left, right = queue.pop(0)
            if not left and right:
                return False
            elif left and not right:
                return False
            elif not (left or right):
                continue
            elif left.val != right.val:
                return False
            else:
                queue.append([left.left, right.left])
                queue.append([left.right, right.right])
        return True
    
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        """ 572.另一棵树的子树 """
        ## 递归 后序遍历, 因为要使用返回的结果判断
        # def backtrack(root, subRoot):
        #     """ 递归函数还是 101.对称二叉树 的逻辑 \n
        #         只能判断以当前节点root和subRoot为根节点的子树是否结构相同 \n """
        #     if not root and subRoot:
        #         return False
        #     elif root and not subRoot:
        #         return False
        #     elif not (root or subRoot):
        #         return True
        #     elif root.val != subRoot.val:
        #         return False
        #     else:
        #         return backtrack(root.left, subRoot.left) and backtrack(root.right, subRoot.right)
        
        # if not subRoot:
        #     return True
        # elif not root:
        #     return False
        # else:
        #     return backtrack(root, subRoot) or self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)

        ## Again 与101.对称二叉树 稍有差别, 并不难, 注意细节
        ## step1.核心逻辑还是判断两棵树是否结构相同
        def backtrack(left, right):
            if not left and right:
                return False
            elif left and not right:
                return False
            elif not (left or right):
                return True
            elif left.val != right.val:
                return False
            else:
                return backtrack(left.left, right.left) and backtrack(left.right, right.right)
        
        ## step2.要同时调用backtrack和self.isSubtree, 所以这里也必须判断(尽管题目给的root和subRoot不空, 但递归下来的可能会空)
        if not subRoot:
            return True
        elif not root:
            return False
        else:
            return backtrack(root, subRoot) or self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)
        
    def maxDepth(self, root: 'Node') -> int:
        """ 559.N叉树的最大深度 """
        from collections import deque

        if not root:
            return 0
        queue = deque([root])
        max_depth = 0
        while queue:
            level_size = len(queue)
            level = []
            for _ in range(level_size):
                node = queue.popleft()
                level.append(node)
                for child in node.children:
                    if child:
                        queue.append(child)
            if level:
                max_depth += 1
        return max_depth
    
    def countNodes(self, root: Optional[TreeNode]) -> int:
        """ 222.完全二叉树的节点个数 """
        ## 前序递归
        # res = 0
        # def backtrack(node):
        #     nonlocal res
        #     if not node:
        #         return
        #     res += 1
        #     backtrack(node.left)
        #     backtrack(node.right)
        
        # backtrack(root)
        # return res

        ## 后序递归
        def backtrack(node):
            if not node:
                return 0
            leftNum = backtrack(node.left)
            rightNum = backtrack(node.right)
            return 1 + leftNum + rightNum
        
        return backtrack(root)

    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        """ 110.平衡二叉树 """
        ## 递归后序
        ## 并不是通过求 最大深度 就能解决的, 需要一点点巧妙
        def backtrack(node):
            """ 返回以node为根节点的平衡二叉树的'高度', 不是平衡二叉树返回-1 """
            if not node:
                return 0
            leftHeight = backtrack(node.left)
            if leftHeight == -1:
                return -1
            rightHeight = backtrack(node.right)
            if rightHeight == -1:
                return -1
            if abs(leftHeight - rightHeight) > 1:
                return -1
            return 1 + max(leftHeight, rightHeight)

        return False if backtrack(root) == -1 else True
    
    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
        """ 257.二叉树的所有路径 \n
            路径, 前序遍历 可以记录父节点指向子节点 """
        # def backtrack(node, path, res):
        #     path.append(node.val)
        #     # 终止条件
        #     if not (node.left or node.right):
        #         res.append('->'.join([str(x) for x in path]))
        #         return
        #     # 递归逻辑
        #     if node.left:
        #         backtrack(node.left, path, res)
        #         path.pop(-1)        # 回溯, 递归和回溯要永远在一起
        #     if node.right:
        #         backtrack(node.right, path, res)
        #         path.pop(-1)        # 回溯
        
        # path = []
        # res = []
        # backtrack(root, path, res)
        # return res

        ## 前序迭代
        # if not root:
        #     return []
        # stack = [root]
        # paths = [[root.val]]     # 对每一个节点, 都会保存路径
        # res = []
        # while stack:
        #     node = stack.pop(-1)
        #     path = paths.pop(-1)
        #     if not (node.left or node.right):
        #         res.append('->'.join([str(x) for x in path]))
        #     if node.right:
        #         stack.append(node.right)
        #         paths.append(path + [node.right.val])
        #     if node.left:
        #         stack.append(node.left)
        #         paths.append(path + [node.left.val])
        # return res

        ## Again 前序迭代
        if not root:
            return []
        stack = [root]          # 栈 遍历节点
        paths = [[root.val]]    # 栈 为每一个节点保存路径. 动作与stack同步即可
        res = []
        while stack:
            node = stack.pop(-1)
            path = paths.pop(-1)
            if not (node.left or node.right):           # 中
                res.append('->'.join([str(x) for x in path]))
            if node.right:                              # 右
                stack.append(node.right)
                paths.append(path + [node.right.val])
            if node.left:                               # 左
                stack.append(node.left)
                paths.append(path + [node.left.val])
        return res

    def sumOfLeftLeaves(self, root: Optional[TreeNode]) -> int:
        """ 404.左叶子之和 \n
            核心: 通过父节点判断左孩子是否为'左叶子' """
        ## 迭代法 更好理解
        stack = [root]
        res = 0
        while stack:
            node = stack.pop(-1)
            if node.left and not (node.left.left or node.left.right):
                res += node.left.val
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return res

        ## 后序递归, 通过求取左右子树结果累加
        # 终止条件
        # if not root:
        #     return 0
        # if not (root.left or root.right):
        #     return 0
        # # 递归逻辑
        # leftVal = self.sumOfLeftLeaves(root.left)
        # if root.left and not (root.left.left or root.left.right):
        #     leftVal = root.left.val
        # rightVal = self.sumOfLeftLeaves(root.right)
        # return leftVal + rightVal
            
    def findBottomLeftValue(self, root: Optional[TreeNode]) -> int:
        """ 513.找树左下角的值 """
        # from collections import deque

        # queue = deque([root])
        # res = 0
        # while queue:
        #     level_size = len(queue)
        #     level = []
        #     for _ in range(level_size):
        #         node = queue.popleft()
        #         level.append(node.val)
        #         if node.left:
        #             queue.append(node.left)
        #         if node.right:
        #             queue.append(node.right)
        #     res = level[0]
        # return res

        ## 递归 虽然更不好理解, 学习下带depth参数
        ## 深度最大 的 叶子节点, 一定是最后一行 --> 前序递归求深度
        ## 保证 左右 遍历顺序, 前中后递归无所谓, 只要是 左右 而不是 右左, 在记录深度最大的叶子节点过程中, 第一个就是 '最左边'
        max_depth = 0       # 更新记录树最大深度
        res = 0
        def backtrack(node, depth):
            nonlocal max_depth, res
            if not (node.left or node.right):
                if depth > max_depth:               # 访问到下一行, 第一个就是'最左边', 更新
                    max_depth = depth
                    res = node.val
            if node.left:
                backtrack(node.left, depth + 1)     # 回溯
            if node.right:
                backtrack(node.right, depth + 1)    # 回溯
        
        backtrack(root, 1)
        return res
    
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        """ 112.路径总和 """
        ## 递归
        # def backtrack(node, residual):
        #     # 终止条件
        #     if not (node.left or node.right) and residual == 0:
        #         return True
        #     elif not (node.left or node.right) and residual != 0:
        #         return False
        #     # 递归逻辑
        #     # 这里可能不好理解, 递归返回的True/False怎么一层一层往上传, 上层又是怎么理解的呢?
        #     # 当遍历到当前节点的时候, 判断没有终止, 发现有左子树--那尝试一下:
        #     #                                   成功 则向上返回True 算法算是结束了
        #     #                                   失败 先不着急返回 尝试其他可能 右子树, 若还是失败, 那向上返回False, 上层节点接受到False转去尝试其他节点
        #     # 通过不断尝试(遍历), 从整个树寻找符合条件的路径
        #     if node.left:
        #         if backtrack(node.left, residual - node.left.val):
        #             return True
        #     if node.right:
        #         if backtrack(node.right, residual - node.right.val):
        #             return True
        #     return False

        # if not root:
        #     return False
        # return backtrack(root, targetSum - root.val)

        ## 迭代 迭代解法不容易想到 推荐上面的递归
        if not root:
            return False
        stack = [[root, root.val]]
        while stack:
            curNode, curSum = stack.pop(-1)
            if not (curNode.left or curNode.right) and curSum == targetSum:
                return True
            if curNode.right:
                stack.append([curNode.right, curSum + curNode.right.val])
            if curNode.left:
                stack.append([curNode.left, curSum + curNode.left.val])
        return False
    
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        """ 113.路径总和II """
        ## 递归
        res = []
        def backtrack(node, path):
            if not (node.left or node.right) and sum(path) == targetSum:
                res.append(path[:])
            if node.left:
                backtrack(node.left, path + [node.left.val])
            if node.right:
                backtrack(node.right, path + [node.right.val])
        
        if not root:
            return []
        backtrack(root, [root.val])
        return res

    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        """ 106.从中序与后序遍历序列构造二叉树 """
        ## 递归
        if len(inorder) == 0:
            return None
        rootVal = postorder[-1]
        root = TreeNode(rootVal)

        ind = inorder.index(rootVal)
        inorder_left = inorder[:ind]
        inorder_right = inorder[ind + 1:]

        postorder_left = postorder[:len(inorder_left)]
        postorder_right = postorder[len(inorder_left):-1]

        root.left = self.buildTree(inorder_left, postorder_left)
        root.right = self.buildTree(inorder_right, postorder_right)
        return root

    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        """ 105. 从前序与中序遍历序列构造二叉树 """
        if len(preorder) == 0:
            return None
        rootVal = preorder[0]
        root = TreeNode(rootVal)

        ind = inorder.index(rootVal)
        inorder_left = inorder[:ind]
        inorder_right = inorder[ind + 1:]

        preorder_left = preorder[1:1 + len(inorder_left)]
        preorder_right = preorder[1 + len(inorder_left):]

        root.left = self.buildTree(preorder_left, inorder_left)
        root.right = self.buildTree(preorder_right, inorder_right)

        return root
    
    def constructMaximumBinaryTree(self, nums: List[int]) -> Optional[TreeNode]:
        """ 654.最大二叉树 """
        # 终止条件
        if len(nums) == 0:
            return None
        # 递归逻辑
        rootVal = max(nums)
        root = TreeNode(rootVal)
        
        ind = nums.index(rootVal)
        root.left = self.constructMaximumBinaryTree(nums[:ind])
        root.right = self.constructMaximumBinaryTree(nums[ind + 1:])
        return root
    
    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
        """ 617.合并二叉树 """
        ## 递归, 但不太好想
        # 终止条件
        # if not root1:
        #     return root2
        # elif not root2:
        #     return root1
        # # 递归调用逻辑
        # root1.val += root2.val
        # root1.left = self.mergeTrees(root1.left, root2.left)
        # root1.right = self.mergeTrees(root1.right, root2.right)
        # return root1

        ## 迭代, 可以回看'对称二叉树', 也同时操作了两个节点
        from collections import deque

        if not root1:
            return root2
        elif not root2:
            return root1
        queue = deque([[root1, root2]])
        while queue:
            node1, node2 = queue.popleft()
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
        ## 递归
        # # 终止条件
        # if not root or root.val == val:
        #     return root
        # # 递归逻辑
        # if val < root.val:
        #     res = self.searchBST(root.left, val)
        # elif val > root.val:
        #     res = self.searchBST(root.right, val)
        # return res

        ## 迭代 如此简洁的迭代
        ## 与普通二叉树不同, 不需要回溯, 因为二叉搜索树的特性已经决定了搜索路径
        while root:
            if val == root.val:
                return root
            elif val < root.val:
                root = root.left
            elif val > root.val:
                root = root.right
        return root
    
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        """ 98.验证二叉搜索树 """
        ## 迭代中序
        # stack = []
        # cur = root
        # preVal = float('-inf')
        # while cur or stack:
        #     if cur:
        #         stack.append(cur)
        #         cur = cur.left
        #     else:
        #         node = stack.pop(-1)
        #         if node.val <= preVal:
        #             return False
        #         preVal = node.val
        #         cur = node.right
        # return True

        ## 递归 你还记得 中序递归 吗
        res = []
        def backtrack(node):
            nonlocal res
            if not node:
                return
            backtrack(node.left)
            res.append(node.val)
            backtrack(node.right)
        backtrack(root)
        for i in range(1, len(res)):
            if res[i - 1] >= res[i]:
                return False
        return True
    
    def getMinimumDifference(self, root: Optional[TreeNode]) -> int:
        """ 530.二叉搜索树的最小绝对差 """
        res = []
        stack = []
        cur = root
        while cur or stack:
            if cur:
                stack.append(cur)
                cur = cur.left
            else:
                node = stack.pop(-1)
                res.append(node.val)
                cur = node.right
        minAbs = float('inf')
        for i in range(1, len(res)):
            minAbs = min(minAbs, abs(res[i - 1] - res[i]))
        return minAbs
    
    def findMode(self, root: Optional[TreeNode]) -> List[int]:
        """ 501.二叉搜索数中的众数 """
        ## 二叉搜索树, 记得 中序遍历
        # maxCount = 0
        # res = []
        # count = 0
        # pre = None
        
        # def backtrack(node):
        #     nonlocal maxCount, res, count, pre
        #     if not node:
        #         return
        #     backtrack(node.left)            # 左
        #     # >>>                             中
        #     if pre is None:
        #         count = 1
        #     elif pre.val == node.val:
        #         count += 1
        #     else:
        #         count = 1
        #     pre = node

        #     if count == maxCount:
        #         res.append(node.val)
        #     elif count > maxCount:
        #         maxCount = count
        #         res.clear()
        #         res.append(node.val)
            
        #     backtrack(node.right)           # 右
        
        # backtrack(root)
        # return res

        ## 中序迭代
        maxCount = 0
        res = []
        count = 0
        pre = None

        cur = root
        stack = []
        while cur or stack:
            if cur:
                stack.append(cur)
                cur = cur.left
            else:
                node = stack.pop(-1)
                # 处理节点
                if pre is None:
                    count = 1
                elif pre.val == node.val:
                    count += 1
                else:
                    count = 1
                pre = node
                if count == maxCount:
                    res.append(node.val)
                elif count > maxCount:
                    maxCount = count
                    res.clear()
                    res.append(node.val)
                
                cur = node.right
        return res
    
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        """ 236.二叉树的最近公共祖先 """
        ## 后序递归, 需要左右子树的返回值来判断是否找到祖先
        # 终止条件 + 本身是祖先
        if not root or root in [p, q]:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        # left right 不空即说明有找到祖先, 那就返回给上层, 谁找到了就把谁返回上去
        if left and right:
            return root
        elif not left and right:
            return right
        elif left and not right:
            return left
        elif not (left or right):
            return None
        
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        """ 235.二叉搜索树的最近公共祖先 \n
            利用BST的性质 从BST的上到下去遍历. 有一个隐藏但很重要的点, 从下往下 若找到祖先, 则就是最近公共祖先, 这一点会让题目变得很简单 \n
            而 236.二叉树的最近公共祖先 普通二叉树则是从下到上回溯遍历 """
        ## 递归
        # # 终止条件
        # if not root:
        #     return
        # # 递归
        # if root.val > p.val and root.val > q.val:
        #     left = self.lowestCommonAncestor(root.left, p, q)
        #     if left:
        #         return left     # 递归遍历一条边, 找到就返回
        # elif root.val < p.val and root.val < q.val:
        #     right = self.lowestCommonAncestor(root.right, p, q)
        #     if right:
        #         return right
        # elif p.val <= root.val <= q.val or q.val <= root.val <= p.val:
        #     return root

        ## 迭代 BST 中序迭代 -- 巨简单的, 还记得不
        while root:
            if root.val < p.val and root.val < q.val:
                root = root.right
            elif root.val > p.val and root.val > q.val:
                root = root.left
            else:
                return root
                

        ## 用普通二叉树的方法, 就要后续遍历了 236.二叉树的最近公共祖先
        # # 终止条件
        # if not root:
        #     return None
        # if root in [p, q]:
        #     return root
        # # 递归调用
        # left = self.lowestCommonAncestor(root.left, p, q)
        # right = self.lowestCommonAncestor(root.right, p, q)
        # if left and right:
        #     return root
        # elif not left and right:
        #     return right
        # elif left and not right:
        #     return left
        # elif not (left or right):
        #     return None
        


if __name__ == '__main__':
    """
    递归三要素:
        1.确定递归函数参数和返回值
        2.终止条件
        3.递归调用逻辑
    """