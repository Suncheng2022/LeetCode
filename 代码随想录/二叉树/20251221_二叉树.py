from typing import Optional, List

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        """ 144.二叉树的前序遍历 """
        ## 递归 前后中统一
        # res = []

        # def backtrack(node):
        #     if not node:
        #         return
        #     res.append(node.val)
        #     backtrack(node.left)
        #     backtrack(node.right)
        
        # backtrack(root)
        # return res

        ## 迭代
        if not root:
            return []
        stack = [root]
        res = []
        while stack:
            node = stack.pop()
            res.append(node.val)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return res
    
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        """ 145.二叉树的后序遍历 """
        ## 递归 前后中统一
        # res = []

        # def backtrack(node):
        #     if not node:
        #         return
        #     backtrack(node.left)
        #     backtrack(node.right)
        #     res.append(node.val)
        
        # backtrack(root)
        # return res

        ## 迭代
        ## 左右中 - 中右左
        if not root:
            return []
        stack = [root]
        res = []
        while stack:
            node = stack.pop()
            res.append(node.val)
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
        return res[::-1]
    
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        """ 94.二叉树的中序遍历 """
        ## 递归 前后中统一
        # res = []

        # def backtrack(node):
        #     if not node:
        #         return
        #     backtrack(node.left)
        #     res.append(node.val)
        #     backtrack(node.right)
        
        # backtrack(root)
        # return res

        ## 迭代
        if not root:
            return []
        res = []
        stack = []
        cur = root
        while cur or stack:
            if cur:
                stack.append(cur)
                cur = cur.left
            else:
                node = stack.pop()
                res.append(node.val)
                cur = node.right
        return res
    
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        """ 102.二叉树的层序遍历 """
        if not root:
            return []
        queue = [root]
        res = []
        while queue:
            num_level = len(queue)
            res_level = []
            for _ in range(num_level):
                node = queue.pop(0)
                res_level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(res_level)
        return res
    
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """ 226.翻转二叉树 """
        # def backtrack(node):
        #     if not node:
        #         return
        #     node.left, node.right = node.right, node.left
        #     backtrack(node.left)
        #     backtrack(node.right)
        
        # backtrack(root)
        # return root

        ## 迭代法
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

    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        """ 101.对称二叉树 """
        # def backtrack(left, right):
        #     if not (left or right):
        #         return True
        #     elif not (left and right):
        #         return False
        #     elif left.val != right.val:
        #         return False
        #     else:
        #         return backtrack(left.left, right.right) and backtrack(left.right, right.left)
        
        # if not root:
        #     return True
        # return backtrack(root.left, root.right)

        ## 迭代法
        if not root:
            return True
        queue = [root.left, root.right]
        while queue:
            left, right = queue.pop(0), queue.pop(0)
            if not (left or right):
                continue
            elif not (left and right):
                return False
            elif left.val != right.val:
                return False
            else:
                queue.extend([left.left, right.right, left.right, right.left])
        return True
    
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        """ 104.二叉树的最大深度 """
        # if not root:
        #     return 0
        # res = 0
        # queue = [root]
        # while queue:
        #     num_level = len(queue)
        #     res_level = []
        #     for _ in range(num_level):
        #         node = queue.pop(0)
        #         res_level.append(node.val)
        #         if node.left:
        #             queue.append(node.left)
        #         if node.right:
        #             queue.append(node.right)
        #     if res_level:
        #         res += 1
        # return res
    
        ## 迭代:
        ##      前序 求深度
        ##      后序 求高度

        # 后序
        # def backtrack(node):
        #     if not node:
        #         return 0
        #     leftHeight = backtrack(node.left)
        #     rightHeight = backtrack(node.right)
        #     return 1 + max(leftHeight, rightHeight)

        # return backtrack(root)

        # 前序
        res = 0

        def backtrack(node, depth):
            nonlocal res
            if not node:
                return 0
            res = max(res, depth)
            if node.left:
                backtrack(node.left, depth + 1)
            if node.right:
                backtrack(node.right, depth + 1)

        backtrack(root, 1)
        return res
    
    def minDepth(self, root: Optional[TreeNode]) -> int:
        """ 111.二叉树的最小深度 """
        ## 递归-前序
        res = float('inf')

        def backtrack(node, depth):
            nonlocal res
            if not (node.left or node.right):
                res = min(res, depth)
                return
            if node.left:
                backtrack(node.left, depth + 1)
            if node.right:
                backtrack(node.right, depth + 1)
        
        if not root:
            return 0
        backtrack(root, 1)
        return res

        ## 递归-后序, 个人觉得不如前序优雅/简单
    def countNodes(self, root: Optional[TreeNode]) -> int:
        """ 222.完全二叉树的节点个数 """
        ## 前序递归
        if not root:
            return 0
        res = 0

        def backtrack(node):
            nonlocal res
            if not node:
                return
            res += 1
            backtrack(node.left)
            backtrack(node.right)
        
        backtrack(root)
        return res
    
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        """ 110.平衡二叉树 """
        if not root:
            return True
        
        def backtrack(node):
            if not node:
                return 0
            leftHight = backtrack(node.left)
            if leftHight == -1:
                return -1
            rightHight = backtrack(node.right)
            if rightHight == -1:
                return -1
            if abs(leftHight - rightHight) > 1:
                return -1
            return 1 + max(leftHight, rightHight)
        
        return backtrack(root) != -1
    
    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
        """ 257.二叉树的所有路径 """
        res = []
        path = []

        def backtrack(node):
            path.append(node.val)
            if not (node.left or node.right):
                res.append('->'.join([str(x) for x in path]))
                return
            if node.left:
                backtrack(node.left)
                path.pop()
            if node.right:
                backtrack(node.right)
                path.pop()
        
        backtrack(root)
        return res
    
    def sumOfLeftLeaves(self, root: Optional[TreeNode]) -> int:
        """ 404.左叶子之和 """
        ## 递归
        # res = 0

        # def backtrack(node):
        #     nonlocal res
        #     if not node:
        #         return
        #     if node.left and not (node.left.left or node.left.right):
        #         res += node.left.val
        #     backtrack(node.left)
        #     backtrack(node.right)
        
        # backtrack(root)
        # return res

        ## 迭代-层序遍历
        res = 0
        queue = [root]
        while queue:
            num_level = len(queue)
            for _ in range(num_level):
                node = queue.pop(0)
                if node.left and not (node.left.left or node.left.right):
                    res += node.left.val
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return res
    
    def findBottomLeftValue(self, root: Optional[TreeNode]) -> int:
        """ 513.找树左下角的值 """
        ## 迭代很好过
        
        ## 递归
        max_depth = 0
        res = 0

        def backtrack(node, depth):
            nonlocal max_depth, res
            if not node:
                return
            if not (node.left or node.right):
                if depth > max_depth:
                    max_depth = depth
                    res = node.val
            backtrack(node.left, depth + 1)
            backtrack(node.right, depth + 1)
        
        backtrack(root, 1)
        return res
    
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        """ 112.路径总和 """
        ## 找一条合法路径, 找到就返回
        ## 递归 其实不简单
        def backtrack(node, count):
            ## 终止条件 就是 遇到叶子节点
            if not (node.left or node.right) and count == 0:
                return True
            elif not (node.left or node.right):
                return False
            
            if node.left:
                if backtrack(node.left, count - node.left.val):
                    return True
            if node.right:
                if backtrack(node.right, count - node.right.val):
                    return True
            return False
        
        if not root:
            return False
        return backtrack(root, targetSum - root.val)
    
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        """ 113.路径总和II """
        res = []
        path = []

        def backtrack(node):
            if not (node.left or node.right):
                if sum(path) == targetSum:
                    res.append(path[:])
                return
            
            if node.left:
                path.append(node.left.val)
                backtrack(node.left)
                path.pop()
            if node.right:
                path.append(node.right.val)
                backtrack(node.right)
                path.pop()
        
        if not root:
            return res
        path.append(root.val)
        backtrack(root)
        return res
    
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        """ 112.路径总和 """
        ## 递归--本题找到合法路径就结束, 所以需要返回值
        def backtrack(node, count):
            ## 终止条件
            if not (node.left or node.right) and count == 0:
                return True
            elif not (node.left or node.right):
                return False
            ## 递归逻辑
            if node.left:
                if backtrack(node.left, count - node.left.val):
                    return True
            if node.right:
                if backtrack(node.right, count - node.right.val):
                    return True
            return False
        
        if not root:
            return False
        return backtrack(root, targetSum - root.val)
    
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        """ 113.路径总和II """
        ## 递归--本题, 需要遍历整棵树, 所以不需要返回值
        res = []
        path = []

        def backtrack(node):
            if not (node.left or node.right):
                if sum(path) == targetSum:
                    res.append(path[:])
                return
            if node.left:
                path.append(node.left.val)
                backtrack(node.left)
                path.pop()
            if node.right:
                path.append(node.right.val)
                backtrack(node.right)
                path.pop()
        
        if not root:
            return []
        path.append(root.val)
        backtrack(root)
        return res
    
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        """ 106.从中序与后序遍历序列构造二叉树 """
        if not (len(inorder) and len(postorder)):
            return
        root = TreeNode(postorder[-1])
        ind_root = inorder.index(postorder[-1])

        root.left = self.buildTree(inorder[:ind_root], postorder[:ind_root])
        root.right = self.buildTree(inorder[ind_root + 1:], postorder[ind_root:-1])
        return root
    
    def constructMaximumBinaryTree(self, nums: List[int]) -> Optional[TreeNode]:
        """ 654.最大二叉树 """
        if not nums:
            return
        maxVal = max(nums)
        maxInd = nums.index(maxVal)
        
        root = TreeNode(maxVal)
        root.left = self.constructMaximumBinaryTree(nums[:maxInd])
        root.right = self.constructMaximumBinaryTree(nums[maxInd + 1:])
        return root
    
    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
        """ 617.合并二叉树 """
        ## 递归-超级简单
        # if not (root1 and root2):
        #     return root2 or root1
        # root1.val += root2.val
        # root1.left = self.mergeTrees(root1.left, root2.left)
        # root1.right = self.mergeTrees(root1.right, root2.right)
        # return root1

        ## 迭代--同时遍历两棵树
        if not (root1 and root2):
            return root1 or root2
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
        ## 迭代--二叉搜索树特性已决定了搜索路径
        # if not root:
        #     return
        # cur = root
        # while cur:
        #     if cur.val == val:
        #         return cur
        #     elif cur.val > val:
        #         cur = cur.left
        #     else:
        #         cur = cur.right
        # return

        ## 递归
        if not root:
            return
        if root.val == val:
            return root
        return self.searchBST(root.left, val) or self.searchBST(root.right, val)
    
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        """ 98.验证二叉搜索树 """
        ## 忘记了 二叉搜索树的中序遍历严格递增
        ## 递归中序
        # curVal = float('-inf')

        # def backtrack(node):
        #     nonlocal curVal
        #     if not node:
        #         return True
            
        #     left = backtrack(node.left)
        #     if node.val <= curVal:
        #         return False
        #     else:
        #         curVal = node.val
        #     right = backtrack(node.right)
        #     return left and right
        
        # return backtrack(root)

        ## 迭代中序
        res = []
        stack = []
        cur = root
        while cur or stack:
            if cur:
                stack.append(cur)
                cur = cur.left
            else:
                node = stack.pop()
                if res and res[-1] >= node.val:
                    return False
                res.append(node.val)
                cur = node.right
        return True
    
    def getMinimumDifference(self, root: Optional[TreeNode]) -> int:
        """ 530.二叉搜索树的最小绝对差 """
        ## 二叉搜索树找最值/差值, 想象在升序数组上操作, 记录上一个节点
        res = float('inf')
        pre = None

        def backtrack(node):
            nonlocal res, pre
            if not node:
                return
            backtrack(node.left)
            if pre is not None:
                res = min(res, node.val - pre.val)      # 升序嘛
            pre = node
            backtrack(node.right)
        
        backtrack(root)
        return res

        ## 这样做不太标准, 标准思路是上面的解法
        # absDiff = float('inf')
        # res = []

        # def backtrack(node):
        #     if not node:
        #         return
        #     nonlocal absDiff, res
        #     backtrack(node.left)
        #     if res and abs(res[-1] - node.val) < absDiff:
        #         absDiff = abs(res[-1] - node.val)
        #     res.append(node.val)
        #     backtrack(node.right)
        
        # backtrack(root)
        # return absDiff

    def findMode(self, root: Optional[TreeNode]) -> List[int]:
        """ 501.二叉搜索数中的众数  """
        ## 递归--二叉搜索树, 中序
        # res = []
        # max_count = 0
        # cur_count = 0
        # pre = None

        # def backtrack(node):
        #     nonlocal max_count, cur_count, pre
        #     if not node:
        #         return
        #     backtrack(node.left)
        #     if pre is None:
        #         cur_count = 1
        #     elif pre.val == node.val:
        #         cur_count += 1
        #     else:
        #         cur_count = 1
        #     pre = node

        #     if cur_count == max_count:
        #         res.append(node.val)
        #     elif cur_count > max_count:
        #         max_count = cur_count
        #         res.clear()
        #         res.append(node.val)
        #     backtrack(node.right)
        
        # backtrack(root)
        # return res

        ## 迭代
        if not root:
            return []
        cur = root
        stack = []
        res = []
        maxCount = 0
        curCount = 0
        pre = None
        while cur or stack:
            if cur:
                stack.append(cur)
                cur = cur.left
            else:
                node = stack.pop()
                if pre is None:
                    curCount = 1
                elif pre.val == node.val:
                    curCount += 1
                else:
                    curCount = 1
                pre = node

                if curCount == maxCount:
                    res.append(node.val)
                elif curCount > maxCount:
                    maxCount = curCount
                    res.clear()
                    res.append(node.val)
                pre = node

                cur = node.right
        return res
    
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        """ 236.二叉树的最近公共祖先 """
        ## 后序遍历, 要通过左右子树的结果判断
        if not root or root in [p, q]:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if left and right:
            return root
        elif not (left and right):
            return left or right