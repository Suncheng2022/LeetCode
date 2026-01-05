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