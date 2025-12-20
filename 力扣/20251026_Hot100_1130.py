from typing import List, Optional

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class MinStack:
    """ 155.最小栈 """
    def __init__(self):
        self.stack = []
        self.minVal = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        if len(self.minVal) == 0:
            self.minVal.append(val)
        else:
            self.minVal.append(min(self.minVal[-1], val))

    def pop(self) -> None:
        _ = self.stack.pop()
        _ = self.minVal.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.minVal[-1]

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        """ 1.两数之和 """
        for i in range(len(nums)):
            if target - nums[i] in nums[i + 1:]:
                return [i, nums.index(target - nums[i], i + 1)]
    
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        """ 160.相交链表 """
        ## 不是快慢指针
        pA = headA
        pB = headB
        while pA != pB:
            pA = pA.next if pA else headB
            pB = pB.next if pB else headA
        return pA   # or pB
    
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        """ 15.三数之和 """
        res = []
        nums.sort()
        for i in range(len(nums) - 2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            l, r = i + 1, len(nums) - 1
            while l < r:
                _tmp = [nums[i], nums[l], nums[r]]
                _sum = sum(_tmp)
                if _sum == 0:
                    res.append(_tmp)
                    while l < r and nums[l + 1] == nums[l]:
                        l += 1
                    while l < r and nums[r - 1] == nums[r]:
                        r -= 1
                    l += 1
                    r -= 1
                elif _sum < 0:
                    l += 1
                else:
                    r -= 1
        return res
    
    def canJump(self, nums: List[int]) -> bool:
        """ 55.跳跃游戏 """
        maxReach = 0
        for i, n in enumerate(nums):
            if maxReach < i:
                return False
            maxReach = max(maxReach, i + n)
        return True
            
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        """ 238.除自己以外数组的乘积 """
        n = len(nums)
        res = [0] * n
        
        tmp = 1
        for i in range(n):
            res[i] = tmp
            tmp *= nums[i]
        tmp = 1
        for i in range(n - 1, -1, -1):
            res[i] *= tmp
            tmp *= nums[i]
        return res
    
    def search(self, nums: List[int], target: int) -> int:
        """ 33.搜索旋转排序数组 """
        l, r = 0, len(nums) - 1
        while l <= r:       # 标准二分查找
            m = (l + r) // 2
            if nums[m] == target:
                return m
            elif nums[m] < nums[r]:             # [m, r]升序
                if nums[m] < target <= nums[r]: # 若target在[m, r]范围
                    l = m + 1
                else:                           # 若target不在[m, r]范围, 收缩查找范围
                    r = m - 1
            else:
                if nums[l] <= target < nums[m]:
                    r = m - 1
                else:
                    l = m + 1
        return -1
    
    def maxProfit(self, prices: List[int]) -> int:
        """ 121.买卖股票的最佳时机 """
        """
        一次买卖, 状态:
            0 不持有
            1 持有
        """
        n = len(prices)
        dp = [[0, 0] for _ in range(n)]
        dp[0] = [0, -prices[0]]
        for i in range(1, n):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
            dp[i][1] = max(-prices[i], dp[i - 1][1])
        return dp[-1][0]
    
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        """ 739.每日温度 """
        ## 递减栈
        res = [0] * len(temperatures)
        stack = []
        for i, t in enumerate(temperatures):
            while stack and t > temperatures[stack[-1]]:    # 如果t相比stack中记录的升高了, 则stack中保存的索引要更新. while可以刷新整个stack, 只要不退出
                ind = stack.pop()
                res[ind] = i - ind
            stack.append(i)     # 体现递减栈--假设没执行到while里面, 则stack记录的温度索引在递减
        return res
    
    def sortColors(self, nums: List[int]) -> None:
        """
        75.颜色分类 \n
        Do not return anything, modify nums in-place instead.
        """
        ## 双指针
        pt0 = pt1 = 0
        for i in range(len(nums)):
            if nums[i] == 1:
                nums[i], nums[pt1] = nums[pt1], nums[i]
                pt1 += 1
            if nums[i] == 0:
                nums[i], nums[pt0] = nums[pt0], nums[i]
                if pt0 < pt1:
                    nums[i], nums[pt1] = nums[pt1], nums[i]
                pt0 += 1
                pt1 += 1

        ## 冒泡
        # n = len(nums)
        # for i in range(n - 1):
        #     flag = False
        #     for j in range(n - 1 - i):
        #         if nums[j] > nums[j + 1]:
        #             nums[j], nums[j + 1] = nums[j + 1], nums[j]
        #             flag = True
        #     if not flag:
        #         break

    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        """ 234.回文链表 """
        def reverseList(head):
            """ 反转链表 """
            _pre = None
            while head:
                _next = head.next
                head.next = _pre        # 第一次执行到这时_pre=None, 链表与前面断开了
                _pre = head
                head = _next            # 当head是最后一个节点, 最后一次执行while, _pre成为第一个节点
            return _pre

        slow = fast = head
        while fast.next and fast.next.next:     # while退出后, fast指向第二段的最后一个节点, slow指向第一段的最后一个节点
            slow = slow.next
            fast = fast.next.next
        head2 = reverseList(slow.next)          # slow.next指向第二段的第一个节点, 执行反转
        while head and head2:                   # 即使长短不同, 都work
            if head.val != head2.val:
                return False
            head = head.next
            head2 = head2.next
        return True

    def productExceptSelf(self, nums: List[int]) -> List[int]:
        """ 238.除自身以外数组的乘积 """
        res = [0] * len(nums)
        
        _tmp = 1
        for i, n in enumerate(nums):
            res[i] = _tmp
            _tmp *= nums[i]
        _tmp = 1
        for i in range(len(nums) - 1, -1, -1):
            res[i] *= _tmp
            _tmp *= nums[i]
        return res
    
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        114.二叉树展开为链表 
        Do not return anything, modify root in-place instead.
        """
        cur = root
        while cur:
            if not cur.left:
                cur = cur.right
            else:
                tmp = cur.right
                cur.right = cur.left
                cur.left = None

                toRight = cur
                while toRight.right:
                    toRight = toRight.right
                toRight.right = tmp

                cur = cur.right

    def canPartition(self, nums: List[int]) -> bool:
        """ 416.分割等和子集 """
        ## 背包--动态规划
        if sum(nums) % 2:
            return False
        target = sum(nums) // 2
        dp = [0] * (target + 1)     # dp[j] 装满容量为j的背包 所得最大价值是多少
        for i in range(len(nums)):
            for j in range(target, nums[i] - 1, -1):    # 一维dp 背包倒序 避免覆盖前面的
                dp[j] = max(dp[j - nums[i]] + nums[i], dp[j])
        return dp[-1] == target
    
    def singleNumber(self, nums: List[int]) -> int:
        """ 136.只出现一次的数字 """
        ## 位运算
        res = nums.pop(0)
        while nums:
            res ^= nums.pop()
        return res

        ## 不够算法
        # nums.sort()
        # stack = []
        # while nums:
        #     n = nums.pop(0)
        #     if not stack or stack[-1] != n:
        #         stack.append(n)
        #     else:
        #         stack.pop()
        # return stack[-1]

    def permute(self, nums: List[int]) -> List[List[int]]:
        """ 46.全排列 """
        ## 回溯三部曲
        res = []
        path = []
        used = [False] * len(nums)

        def backtrack():
            if len(path) == len(nums):
                res.append(path[:])
                return
            for i in range(len(nums)):
                if used[i] == True:
                    continue
                used[i] = True
                path.append(nums[i])
                backtrack()
                path.pop()
                used[i] = False
        
        backtrack()
        return res
    
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        """ 240.搜索二维矩阵II """
        ## 利用matrix的特性
        m, n = len(matrix), len(matrix[0])
        rowInd, colInd = 0, n - 1       # 起点
        while rowInd < m and colInd >= 0:
            val = matrix[rowInd][colInd]
            if val == target:
                return True
            elif val < target:
                rowInd += 1
            else:
                colInd -= 1
        return False

        ## 不够算法 没有利用matrix特性
        # def binarySearch(nums):
        #     l, r = 0, len(nums) - 1
        #     while l <= r:
        #         m = (l + r) // 2
        #         if nums[m] == target:
        #             return True
        #         elif nums[m] > target:
        #             r = m - 1
        #         else:
        #             l = m + 1
        #     return False
        
        # for row in matrix:
        #     if binarySearch(row):
        #         return True
        # return False

    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        """ 739.每日温度 """
        ## 递减栈, 参考之前
        answers = [0] * len(temperatures)
        stack = []
        for i, t in enumerate(temperatures):
            while stack and t > temperatures[stack[-1]]:
                ind = stack.pop()
                answers[ind] = i - ind
            stack.append(i)     # 若没执行while, 就能理解这是递减栈了
        return answers
    
    def longestConsecutive(self, nums: List[int]) -> int:
        """ 128.最长连续序列 """
        ## 动规, 能做 但复杂度不符合要求
        # if len(nums) == 0:
        #     return 0
        # nums = list(set(nums))
        # nums.sort()
        # n = len(nums)
        # dp = [1] * n        # dp[i] 以nums[i]结尾的最长连续序列的长度
        # for i in range(1, n):
        #     if nums[i - 1] + 1 == nums[i]:
        #         dp[i] = dp[i - 1] + 1
        # return max(dp)

        ## 建议这个解法-->参考之前
        nums = set(nums)
        res = 0
        for dg in nums:
            if dg - 1 not in nums:
                _len = 1
                while dg + 1 in nums:
                    _len += 1
                    dg += 1
                res = max(res, _len)
        return res
    
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        """ 207.课程表 """
        ## 还是参考之前, 忘了
        from collections import defaultdict

        res = 0
        i2indegree = [0] * numCourses
        p2i = defaultdict(set)
        for cur, pre in prerequisites:
            i2indegree[cur] += 1
            p2i[pre].add(cur)
        queue = [i for i, indegree in enumerate(i2indegree) if indegree == 0]

        while queue:
            cur = queue.pop(0)
            res += 1
            for ind in p2i[cur]:
                i2indegree[ind] = max(0, i2indegree[ind] - 1)
                if i2indegree[ind] == 0:
                    queue.append(ind)
        return res == numCourses
        
    def sortColors(self, nums: List[int]) -> None:
        """
        75.颜色分类 \n
        Do not return anything, modify nums in-place instead.
        """
        ## 一趟扫描
        pos0 = pos1 = 0
        for i in range(len(nums)):
            if nums[i] == 1:
                nums[i], nums[pos1] = nums[pos1], nums[i]
                pos1 += 1
            
            if nums[i] == 0:
                nums[i], nums[pos0] = nums[pos0], nums[i]
                pos0 += 1
                if nums[i] == 1:
                    nums[i], nums[pos1] = nums[pos1], nums[i]
                pos1 += 1

        ## 冒泡
        # n = len(nums)
        # for i in range(n - 1):
        #     flag = False
        #     for j in range(n - 1 - i):
        #         if nums[j] > nums[j + 1]:
        #             nums[j], nums[j + 1] = nums[j + 1], nums[j]
        #             flag = True
        #     if not flag:
        #         break
        
    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
        """ 617.合并二叉树 """
        from collections import deque

        if not (root1 and root2):
            return root1 or root2
        
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
    
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """ 148.排序链表 """
        ## 归排
        def mysplit(head, step):
            """ 将链表head划分为两段: 长度为step + 长度为剩下, 返回第二段的第一个节点 """
            for _ in range(step - 1):
                if not head:
                    return
                head = head.next
            if head:
                res = head.next
                head.next = None
                return res
            else:
                return
        
        def mymerge(h1, h2):
            cur = dummyHead = ListNode()
            while h1 and h2:
                if h1.val < h2.val:
                    cur.next = h1
                    h1 = h1.next
                else:
                    cur.next = h2
                    h2 = h2.next
                cur = cur.next
            cur.next = h1 or h2
            return dummyHead.next
        
        if not (head and head.next):
            return head

        n = 0
        cur = head
        while cur:
            n += 1
            cur = cur.next
        
        dummyHead = ListNode(next=head)
        
        step = 1
        while step < n:
            ## 一次归排
            prev = dummyHead
            cur = prev.next
            while cur:
                left = cur
                right = mysplit(left, step)
                cur = mysplit(right, step)

                prev.next = mymerge(left, right)
                while prev.next:
                    prev = prev.next

            step *= 2
        return dummyHead.next
    
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        """ 399.除法求值 """
        graph = {}
        for [s, t], v in zip(equations, values):
            if s not in graph:
                graph[s] = {t: v}
            else:
                graph[s][t] = v
            if t not in graph:
                graph[t] = {s: 1 / v}
            else:
                graph[t][s] = 1 / v
        
        def dfs(s, t):
            if s not in graph:
                return -1
            elif s == t:
                return 1
            for node in graph[s].keys():
                if node == t:
                    return graph[s][t]
                elif node not in visited:
                    visited.add(node)
                    v = dfs(node, t)
                    if v != -1:
                        return v * graph[s][node]
            return -1

        res = []
        for s, t in queries:
            visited = set()
            v = dfs(s, t)
            res.append(v)
        return res

if __name__ == '__main__':
    sl = Solution()

    numCourses = 2
    prerequisites = [[1,0],[0,1]]
    print(sl.canFinish(numCourses, prerequisites))