'''
10.26   执行力这么差?
'''
from typing import List, Optional
import math

# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class LRUCache:
    """ 146.LRU缓存 """
    def __init__(self, capacity: int):
        from collections import OrderedDict

        self.cache = OrderedDict()
        self.capcity = capacity

    def get(self, key: int) -> int:
        if key in self.cache:
            self.cache.move_to_end(key)
        return self.cache.get(key, -1)

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capcity:
            self.cache.popitem(last=False)

class TireNode:
    def __init__(self):
        from collections import defaultdict

        self.children = defaultdict(TireNode)
        self.isword = False

class Trie:
    """ 208.实现Tire(前缀树) """
    def __init__(self):
        self.root = TireNode()

    def insert(self, word: str) -> None:
        cur = self.root
        for c in word:
            cur = cur.children[c]
        cur.isword = True

    def search(self, word: str) -> bool:
        cur = self.root
        for c in word:
            cur = cur.children.get(c, None)
            if cur is None:
                return False
        return cur.isword

    def startsWith(self, prefix: str) -> bool:
        cur = self.root
        for c in prefix:
            cur = cur.children.get(c, None)
            if cur is None:
                return False
        return True
    
class MinStack:
    """ 155.最小栈 """
    def __init__(self):
        self.stack = [0]
        self.minVal = [float('inf')]

    def push(self, val: int) -> None:
        self.minVal.append(min(self.minVal[-1], val))
        self.stack.append(val)

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
    
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        """ 49.字母异位词分组 """
        from collections import defaultdict

        mydict = defaultdict(list)
        for word in strs:
            k = ''.join(sorted(word))
            mydict[k].append(word)
        return list(mydict.values())
    
    def longestConsecutive(self, nums: List[int]) -> int:
        """ 128.最长连续序列 """
        nums = set(nums)
        res = 0
        for dg in nums:
            if dg - 1 not in nums:      # 保证dg是序列开头
                _len = 1
                while dg + 1 in nums:
                    _len += 1
                    dg += 1
                res = max(res, _len)
        return res

    def moveZeroes(self, nums: List[int]) -> None:
        """
        283.移动零 \n
        Do not return anything, modify nums in-place instead.
        """
        ## 指定放置非零元素的位置.
        ## 不会发生覆盖问题
        ind = 0
        for i in range(len(nums)):      # i比ind要快
            if nums[i] != 0:
                nums[ind] = nums[i]
                ind += 1
        for i in range(ind, len(nums)):
            nums[i] = 0

        ## 将0后移 或 0前移 的想法, 走不通, 解决不了相邻元素为零/非零的情况

    def maxArea(self, height: List[int]) -> int:
        """ 11.盛最多水的容器 """
        l, r = 0, len(height) - 1
        maxArea = (r - l) * min(height[l], height[r])
        while l < r:
            if height[l] < height[r]:       # 要移动矮的柱子, 这样才有机会容量更大. (若移动较高的柱子, 则最矮的柱子不会变, 同时宽度在减小, 所以不可能出现更大容量)
                l += 1
            else:
                r -= 1
            curArea = (r - l) * min(height[l], height[r])
            maxArea = max(maxArea, curArea)
        return maxArea
    
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        """ 15.三数之和 """
        nums.sort()
        res = []
        for i in range(len(nums) - 2):
            if i > 0 and nums[i] == nums[i - 1]:                # 解超时
                continue
            l, r = i + 1, len(nums) - 1
            while l < r:
                _tmp = [nums[i], nums[l], nums[r]]
                _sum = sum(_tmp)
                if _sum == 0:
                    res.append(_tmp)                            # 前面跳过重复计算了, 那就不会有重复结果了, 可以直接加入
                    while l < r and nums[l] == nums[l + 1]:     # 解超时
                        l += 1
                    while l < r and nums[r] == nums[r - 1]:     # 解超时
                        r -= 1
                    l += 1
                    r -= 1
                elif _sum < 0:
                    l += 1
                else:
                    r -= 1
        return res
    
    def lengthOfLongestSubstring(self, s: str) -> int:
        """ 3.无重复字符的最长子串 \n
            要求连续 """
        ## 高效 + 边界更容易考虑
        # maxLength = 0
        # for i in range(len(s)):
        #     seen = set()
        #     for j in range(i, len(s)):
        #         if s[j] in seen:
        #             break
        #         seen.add(s[j])
        #     maxLength = max(maxLength, len(seen))
        # return maxLength

        ## 低效, 且考虑边界的方式复杂
        # if len(s) <= 1:
        #     return len(s)
        # maxLength = 0
        # for i in range(len(s)):
        #     _cur = s[i]
        #     for j in range(i + 1, len(s)):
        #         if s[j] in _cur:
        #             maxLength = max(maxLength, len(_cur))
        #             break
        #         else:
        #             _cur += s[j]
        #             maxLength = max(maxLength, len(_cur))
        # return maxLength

        ## Again, 重复上面高效的实现
        maxLength = 0
        for i in range(len(s)):
            seen = set()            # 滑动窗口 + set查找高效
            for j in range(i, len(s)):
                if s[j] in seen:
                    break
                seen.add(s[j])
            maxLength = max(maxLength, len(seen))
        return maxLength
        
    def findAnagrams(self, s: str, p: str) -> List[int]:
        """ 438.找到字符串中所有字母异位词 """
        ## 参考之前的实现
        res = []
        n = len(s)
        m = len(p)
        if n < m:
            return res
        
        # 初始化滑窗
        s_count = [0] * 26
        p_count = [0] * 26
        for i in range(m):      
            s_count[ord(s[i]) - ord('a')] += 1
            p_count[ord(p[i]) - ord('a')] += 1
        
        if s_count == p_count:
            res.append(0)
        for i in range(m, n):
            s_count[ord(s[i]) - ord('a')] += 1      # 进滑窗
            s_count[ord(s[i - m]) - ord('a')] -= 1  # 出滑窗
            if s_count == p_count:
                res.append(i - m + 1)               # 出滑窗元素的索引是i-m, 则当前滑窗起始索引是i-m+1
        return res

        ## 能过, 但效率太低
        # res = []
        # m = len(p)
        # _p = ''.join(sorted(p))
        # for i in range(0, len(s) - m + 1):
        #     _s = ''.join(sorted(s[i:i + m]))
        #     if _s == _p:
        #         res.append(i)
        # return res

    def subarraySum(self, nums: List[int], k: int) -> int:
        """ 560.和为K的子数组 \n
            要求连续 """
        ## 优化效率--前缀和, 记忆累加结果, 计算结果重复利用
        from collections import defaultdict

        preSums = defaultdict(int)      # 前缀和: 前缀和出现的次数
        preSums[0] = 1                  # 初始化, 也很重要
        presum = 0                      # 当前的前缀和
        res = 0                         # 最终结果
        
        for i in range(len(nums)):
            presum += nums[i]
            res += preSums[presum - k]
            preSums[presum] += 1
        return res

        ## 超时
        # res = 0
        # for i in range(len(nums)):
        #     for j in range(i, len(nums)):
        #         if sum(nums[i:j + 1]) == k:
        #             res += 1
        # return res

    def maxSubArray(self, nums: List[int]) -> int:
        """ 53.最大子数组和 \n
            要求连续 """
        n = len(nums)
        dp = [0] * n    # dp[i] 以nums[i]结尾的最大子数组和
        dp[0] = nums[0]
        for i in range(1, n):
            dp[i] = max(dp[i - 1] + nums[i], nums[i])
        return max(dp)

    
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        """ 56.合并区间 """
        ## 其实很简单哈
        intervals.sort(key=lambda x: x[0])
        res = [intervals[0]]
        for inter in intervals[1:]:
            if res[-1][-1] >= inter[0]:
                res[-1][-1] = max(res[-1][-1], inter[1])
            else:
                res.append(inter)
        return res
    
    def rotate(self, nums: List[int], k: int) -> None:
        """
        189.轮转数组 \n
        Do not return anything, modify nums in-place instead.
        """
        # nums = A + B
        # 1.整个序列反转 rev(nums) = rev(B) + rev(A)
        # 2.单独对B反转 得 rev(rev(B)) + rev(A)
        # 3.单独对A反转 得 rev(rev(B)) + rev(rev(A)) = B + A, 即为所求
        def rev(i, j):
            while i < j:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
                j -= 1
        
        n = len(nums)
        k %= n
        rev(0, n - 1)
        rev(0, k - 1)
        rev(k, n - 1)

    def productExceptSelf(self, nums: List[int]) -> List[int]:
        """ 238.除自身以外数组的乘积 """
        # 正着乘一遍, 反着乘一遍
        n = len(nums)
        res = [0] * n       # 保存结果
        tmp = 1
        for i in range(n):
            res[i] = tmp    # 先赋值
            tmp *= nums[i]  # 再累乘
        tmp = 1
        for i in range(n - 1, -1, -1):
            res[i] *= tmp
            tmp *= nums[i]
        return res
    
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        73.矩阵置零 \n
        Do not return anything, modify matrix in-place instead.
        """
        zero_rows = []
        zero_cols = []
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == 0:
                    zero_rows.append(i)
                    zero_cols.append(j)
        for i in zero_rows:
            matrix[i] = [0] * len(matrix[0])
        for j in zero_cols:
            for i in range(len(matrix)):
                matrix[i][j] = 0
        
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        """ 54.螺旋矩阵 """
        l, r = 0, len(matrix[0]) - 1
        t, b = 0, len(matrix) - 1
        res = []
        while True:
            for i in range(l, r + 1):
                res.append(matrix[t][i])
            t += 1
            if t > b:
                break

            for i in range(t, b + 1):
                res.append(matrix[i][r])
            r -= 1
            if l > r:
                break

            for i in range(r, l - 1, -1):
                res.append(matrix[b][i])
            b -= 1
            if t > b:
                break

            for i in range(b, t - 1, -1):
                res.append(matrix[i][l])
            l += 1
            if l > r:
                break
        return res
    
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        48.旋转图像 \n
        Do not return anything, modify matrix in-place instead.
        """

        ## 顺时针转90° = 转置 + 行反转
        n = len(matrix)
        for i in range(n):
            for j in range(i):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        for i in range(n):
            matrix[i] = matrix[i][::-1]     # 列表的reverse()是原地操作

        ## not in-place
        # n = len(matrix)
        # res = [[0] * n for _ in range(n)]
        # for i in range(n):
        #     for j in range(n):
        #         res[j][n - 1 - i] = matrix[i][j]    # i,j是matrix的, 更好理解
        # matrix[:] = res

    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        """ 240.搜索二维矩阵II """
        def binarySearch(nums):
            l, r = 0, len(nums) - 1
            while l <= r:
                m = (l + r) // 2
                if nums[m] == target:
                    return True
                elif nums[m] > target:
                    r = m - 1
                else:
                    l = m + 1
            return False
        
        for row in matrix:
            if binarySearch(row):
                return True
        return False
    
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        """ 160.相交链表 """
        pA, pB = headA, headB
        while pA != pB:
            pA = pA.next if pA else headB       # pA走完A, 就开始走B. 设A的长度=a+c
            pB = pB.next if pB else headA       # pB走完B, 就开始走A. 设B的长度=b+c. 这个步骤相当于让pA pB都走a+b+c步, 总会相遇
        return pA
        
        ## 自己想到两层循环遍历 超时

    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """ 206.反转链表 """
        pre = None
        cur = head
        while cur:
            next = cur.next
            cur.next = pre
            pre = cur
            cur = next
        return pre

        ## 自己写的能过, 但感觉不太优雅
        # if not head:
        #     return head
        
        # stack = []
        # pH = head
        # while pH:
        #     stack.append(pH)
        #     pH = pH.next
        # pH = stack.pop()
        # res = pH
        # while stack:
        #     node = stack.pop()
        #     pH.next = node
        #     pH = node
        # pH.next = None
        # return res

    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        """ 234.回文链表 """
        ## 思路简单, 不够优雅
        # stack = []
        # while head:
        #     stack.append(head.val)
        #     head = head.next
        
        # l, r = 0, len(stack) - 1
        # while l <= r:
        #     if stack[l] != stack[r]:
        #         return False
        #     l += 1
        #     r -= 1
        # return True
        
        ## 借鉴 206.反转链表
        def reverse_link(head):
            pre = None
            while head:
                next = head.next
                head.next = pre
                pre = head
                head = next
            return pre

        slow = fast = head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        
        head2 = reverse_link(slow.next)
        while head2:
            if head.val != head2.val:
                return False
            head = head.next
            head2 = head2.next
        return True

    def hasCycle(self, head: Optional[ListNode]) -> bool:
        """ 141.环形链表 """
        ## 快慢指针
        slow = fast = head
        while fast:
            slow = slow.next
            fast = fast.next
            if fast:
                fast = fast.next
            
            if slow == fast:
                break
        if not fast:
            return False
        return True

        ## 没超时, 但效率不高
        # nodes = set()
        # while head is not None:
        #     if head in nodes:
        #         return True
        #     nodes.add(head)
        #     head = head.next
        # return False

    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """ 142.环形链表II """
        ## Again, 快慢指针
        slow = fast = head
        while fast:
            slow = slow.next
            fast = fast.next
            if fast:
                fast = fast.next
            
            if slow == fast:
                break
        if fast is None:        # 没有环
            return None
        fast = head
        while slow != fast:
            slow = slow.next
            fast = fast.next
        return slow             # or return fast

        ## 快慢指针
        # slow = fast = head
        # while fast:
        #     slow = slow.next
        #     fast = fast.next
        #     if fast:
        #         fast = fast.next
            
        #     if slow == fast:        # 相遇点
        #         break
        
        # if fast is None:            # 从while跳出, 没有环
        #     return None
        # fast = head                 # 从slow==fast跳出, 此时slow在相遇点
        # while slow != fast:         # 注:可能不会执行, 比如入口就是head
        #     slow = slow.next
        #     fast = fast.next
        # return slow

    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        """ 21.合并两个有序链表 """
        ## 自己还是低效, 答案吧~
        resHead = tmp = ListNode()
        while list1 and list2:
            if list1.val < list2.val:
                tmp.next = list1
                list1 = list1.next
            else:
                tmp.next = list2
                list2 = list2.next
            tmp = tmp.next
        tmp.next = list1 if list1 else list2
        return resHead.next

    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        """ 2.两数相加 """
        cur = head = ListNode()
        sum, mt10 = 0, False        # 当前位置节点的和; more than 10
        while l1 or l2:
            sum = 0
            if l1:
                sum += l1.val
                l1 = l1.next
            if l2:
                sum += l2.val
                l2 = l2.next
            if mt10 is True:
                sum += 1
            cur.next = ListNode(sum % 10)
            cur = cur.next
            mt10 = sum >= 10        # 是否进位, 体现在下一位
        
        if mt10 is True:
            cur.next = ListNode(1)
        return head.next

        ## 效率较低
        # num1 = 0
        # i = 0
        # while l1:
        #     num1 = l1.val * 10 ** i + num1
        #     l1 = l1.next
        #     i += 1
        # num2 = 0
        # i = 0
        # while l2:
        #     num2 = l2.val * 10 ** i + num2
        #     l2 = l2.next
        #     i += 1
        # res = num1 + num2
        # head = ListNode()
        # if res == 0:
        #     return head
        # cur = head
        # while res:
        #     cur.next = ListNode(res % 10)
        #     cur = cur.next
        #     res //= 10
        # return head.next

    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        """ 19.删除链表的倒数第N个节点 """
        ## 头结点, 简约~
        dummyHead = ListNode()
        dummyHead.next = head
        pre = dummyHead
        front = end = head

        for _ in range(n):
            end = end.next
        while end:
            pre = front
            front = front.next
            end = end.next
        pre.next = front.next
        return dummyHead.next
    
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """ 24.两两交换链表中的节点 \n
            就是每两个一组交换, 奇数的话最后一个不动, 别想复杂 """
        ## 头结点~简约
        dummyHead = ListNode()
        dummyHead.next = head
        prev = dummyHead

        while head and head.next:
            first = head
            second = head.next

            prev.next = second
            first.next = second.next
            second.next = first

            prev = first
            head = first.next
        return dummyHead.next
        
        ## 不太容易理解, 边界条件也不好考虑
        # if not head or not head.next:       # 边界条件, 处理0或1个节点的情况
        #     return head
        # cur = head
        # stack = [cur, cur.next]
        # cur = cur.next.next                 # 指向第一个未被访问的节点

        # head = stack.pop()
        # head.next = stack.pop()
        # pre = head.next                     # 指向最后一个被访问的节点

        # while cur and cur.next:
        #     stack.extend([cur, cur.next])
        #     cur = cur.next.next

        #     pre.next = stack.pop()
        #     pre.next.next = stack.pop()
        #     pre = pre.next.next

        # if cur and not cur.next:            # 奇数个节点的情况, 处理最后一个节点
        #     pre.next = cur
        #     pre = pre.next
        # pre.next = None                     # pre只想最后一个被访问的节点, 那必然处理一下next指针, 否则可能死循环
        # return head

    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        """ 138.随机链表的复制 """
        # 参考: https://leetcode.cn/problems/copy-list-with-random-pointer/submissions/675733996/?envType=study-plan-v2&envId=top-100-liked
        ## 拼接 + 拆分, 空间O(1)
        if not head:
            return None
        cur = head
        while cur:
            tmp = Node(cur.val)
            tmp.next = cur.next
            cur.next = tmp
            cur = cur.next.next
        
        cur = head
        while cur:
            if cur.random:
                cur.next.random = cur.random.next
            cur = cur.next.next

        cur = res = head.next
        while cur.next:
            cur.next = cur.next.next
            cur = cur.next
        cur.next = None
        return res
        
        ## 哈希, 两次遍历, 空间O(N)
        # if not head:
        #     return None
        # old2new = {}        # 新旧节点map
        # cur = head
        # while cur:
        #     old2new[cur] = Node(cur.val)
        #     cur = cur.next
        
        # cur = head
        # while cur:
        #     old2new[cur].next = old2new.get(cur.next, None)     # next 可能空
        #     old2new[cur].random = old2new.get(cur.random, None) # random 可能空
        #     cur = cur.next
        # return old2new[head]
        
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """ 148.排序链表 """
        def myMerge(l1, l2):
            cur = dummy = ListNode()
            while l1 and l2:
                if l1.val < l2.val:
                    cur.next = l1
                    l1 = l1.next
                else:
                    cur.next = l2
                    l2 = l2.next
                cur = cur.next
            cur.next = l1 or l2
            return dummy.next
        
        def mySplit(head, k):
            """ 将链表k划分为: k个节点 + 剩余部分 \n
                返回 剩余部分的第一个节点 """
            ## 划分k个节点
            for _ in range(k - 1):      # 细节 k - 1
                if not head:
                    return
                head = head.next
            if not head:
                return
            ## 剩余部分
            right = head.next
            head.next = None
            return right

        if not head or not head.next:
            return head
        
        n = 0
        cur = head
        while cur:
            n += 1
            cur = cur.next
        
        dummyHead = ListNode()
        dummyHead.next = head

        step = 1
        while step < n:
            prev = dummyHead
            cur = prev.next
            
            while cur:                      # 一次归并 
                left = cur
                right = mySplit(left, step)
                cur = mySplit(right, step)
                
                mergeNode = myMerge(left, right)
                prev.next = mergeNode
                while prev.next:
                    prev = prev.next

            step *= 2
        return dummyHead.next
    
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        """ 94.二叉树的中序遍历 """
        ## 递归
        res = []
        if not root:
            return res
        
        def func(node):
            if not node:
                return
            func(node.left)
            res.append(node.val)
            func(node.right)
        
        func(root)
        return res

        ## 栈
        # if not root:
        #     return []
        # res = []
        # stack = []
        # cur = root
        # while stack or cur:
        #     if cur:
        #         stack.append(cur)
        #         cur = cur.left
        #     else:
        #         cur = stack.pop()
        #         res.append(cur.val)
        #         cur = cur.right
        # return res

    def maxDepth(self, root: Optional[TreeNode]) -> int:
        """ 104.二叉树的最大深度 """
        ## 层序遍历, 首选方案
        # if not root:
        #     return 0
        # res = []
        # stack = [root]
        # while stack:
        #     num_level = len(stack)
        #     res_level = []
        #     for _ in range(num_level):
        #         node = stack.pop(0)
        #         res_level.append(node.val)
        #         if node.left:
        #             stack.append(node.left)
        #         if node.right:
        #             stack.append(node.right)
        #     res.append(res_level[:])
        # return len(res)

        ## 递归
        if not root:
            return 0
        max_depth = 0
        def func(node, depth):
            if not node:
                return
            nonlocal max_depth
            max_depth = max(max_depth, depth)
            if node.left:
                func(node.left, depth + 1)
            if node.right:
                func(node.right, depth + 1)
        
        func(root, 1)
        return max_depth

    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """ 226.翻转二叉树 """
        ## 层序遍历
        # if not root:
        #     return None
        # stack = [root]
        # while stack:
        #     num_level = len(stack)
        #     for _ in range(num_level):
        #         node = stack.pop(0)
        #         node.left, node.right = node.right, node.left
        #         if node.left:
        #             stack.append(node.left)
        #         if node.right:
        #             stack.append(node.right)
        # return root

        ## 递归
        if not root:
            return None
        
        def func(node):
            if not node:
                return
            node.left, node.right = node.right, node.left
            func(node.left)
            func(node.right)
        
        func(root)
        return root
    
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        """ 101.对称二叉树 """
        ## 递归 后序-->要点:想到参数是两个节点哈哈
        # def func(h1, h2):
        #     if not (h1 or h2):
        #         return True
        #     elif not (h1 and h2):
        #         return False
        #     elif h1.val != h2.val:
        #         return False
        #     else:
        #         leftRes = func(h1.left, h2.right)
        #         rightRes = func(h1.right, h2.left)
        #         return leftRes and rightRes
        # return func(root.left, root.right)

        ## 迭代, 代码几乎同递归
        queue = [[root.left, root.right]]
        while queue:
            h1, h2 = queue.pop(0)
            if not (h1 or h2):
                continue
            elif not (h1 and h2):
                return False
            elif h1.val != h2.val:
                return False
            else:
                queue.append([h1.left, h2.right])
                queue.append([h1.right, h2.left])
        return True
    
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        """ 543.二叉树的直径 """
        res = 0
        def func(node):
            """ 以node为根节点的树的高度 """
            if not node:
                return 0
            nonlocal res
            l = func(node.left)
            r = func(node.right)
            res = max(res, l + r)
            return max(l, r) + 1
        func(root)
        return res
    
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        """ 102.二叉树的层序遍历 """
        if not root:
            return []
        res = []
        queue = [root]
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
    
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        """ 108.将有序数组转化为二叉搜索树 """
        def func(nums):
            if len(nums) == 0:
                return
            i, j = 0, len(nums) - 1
            mid = (i + j) // 2
            root = TreeNode(nums[mid])
            root.left = func(nums[i:mid])
            root.right = func(nums[mid + 1:])
            return root
        return func(nums)

    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        """ 98.验证二叉搜索树 """
        ## 中序遍历, 遍历结果应当升序
        # res = []
        # stack = []
        # cur = root
        # while cur or stack:
        #     if cur:
        #         stack.append(cur)
        #         cur = cur.left
        #     else:
        #         node = stack.pop()
        #         res.append(node.val)
        #         cur = node.right
        # for i in range(1, len(res)):
        #     if res[i - 1] >= res[i]:
        #         return False
        # return True

        ## 递归 中序
        res = []
        def func(node):
            if not node:
                return
            func(node.left)
            res.append(node.val)
            func(node.right)
        
        func(root)
        for i in range(1, len(res)):
            if res[i - 1] >= res[i]:
                return False
        return True

    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        """ 230.二叉搜索树中第K小的元素 """
        ## 中序, 遍历到第k个元素直接返回
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
                if len(res) == k:
                    return res[-1]
                cur = node.right

    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        """ 199.二叉树的右视图 """
        ## 层序遍历
        if not root:
            return []
        res = []
        queue = [root]
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
            res.append(res_level[-1])
        return res
    
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        114.二叉树展开为链表 \n
        Do not return anything, modify root in-place instead.
        """
        if not root:
            return
        cur = root
        while cur:
            if not cur.left:
                cur = cur.right
                continue
            else:
                tmp = cur.right
                cur.right = cur.left
                cur.left = None

                _cur = cur
                while _cur.right:
                    _cur = _cur.right
                _cur.right = tmp

                cur = cur.right
        return root
    
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        """ 105.从前序与中序遍历序列构造二叉树 """
        def func(preorder, inorder):
            if len(preorder) == 0:
                return
            rootVal = preorder[0]
            root = TreeNode(rootVal)
            ind = inorder.index(rootVal)        # 中序索引
            root.left = func(preorder[1:ind + 1], inorder[:ind])
            root.right = func(preorder[ind + 1:], inorder[ind + 1:])
            return root
        
        return func(preorder, inorder)
                
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        """ 437.路径总和III """
        prefixSumMap = {0: 1}
        def func(node, curSum):
            if not node:
                return 0
            res = 0
            curSum += node.val
            res += prefixSumMap.get(curSum - targetSum, 0)

            prefixSumMap[curSum] = prefixSumMap.get(curSum, 0) + 1
            res += func(node.left, curSum)
            res += func(node.right, curSum)
            prefixSumMap[curSum] -= 1
            return res
        
        return func(root, 0)
    
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        """ 236.二叉树的最近公共祖先 \n
            递归思路要清楚: 在以root为根的子树中找p或q, 如果p或q都存在则返回最近公共祖先, 若只有一个在则返回自己, 若都不在则返回None """
        if not root or root in [p, q]:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)       # 左子树是否包含p或q
        right = self.lowestCommonAncestor(root.right, p, q)     # 右子树是否包含p或q
        if left and right:
            return root
        elif not left and right:
            return right
        elif left and not right:
            return left
        elif not (left or right):
            return None
        
    def numIslands(self, grid: List[List[str]]) -> int:
        """ 200.岛屿数量 """
        m = len(grid)
        n = len(grid[0])
        visited = [[False] * n for _ in range(m)]
        res = 0

        def dfs(x, y):
            """ 标记所有与[x,y]相连的陆地 """
            visited[x][y] = True
            for dc in [-1, 0], [1, 0], [0, -1], [0, 1]:
                nextX = x + dc[0]
                nextY = y + dc[1]
                if nextX < 0 or nextX >= m or nextY < 0 or nextY >= n:
                    continue
                if visited[nextX][nextY] == False and grid[nextX][nextY] == '1':
                    dfs(nextX, nextY)
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1' and visited[i][j] == False:
                    res += 1
                    dfs(i, j)
        return res
    
    def orangesRotting(self, grid: List[List[int]]) -> int:
        """ 994. 腐烂的橘子 \n
            与 200.岛屿数量 稍稍相似, 彼为dfs, 此为bfs """
        m = len(grid)
        n = len(grid[0])
        
        round = 0
        count = 0   # 新鲜橘子数量
        queue = []  # 腐烂橘子位置
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    count += 1
                elif grid[i][j] == 2:
                    queue.append([i, j])
        
        while count > 0 and len(queue) > 0:
            round += 1
            num_bad = len(queue)
            for _ in range(num_bad):
                x, y = queue.pop(0)
                for dir in [-1, 0], [1, 0], [0, -1], [0, 1]:
                    nextX, nextY = x + dir[0], y + dir[1]
                    if nextX < 0 or nextX >= m or nextY < 0 or nextY >= n:
                        continue
                    if grid[nextX][nextY] == 1:
                        grid[nextX][nextY] = 2
                        queue.append([nextX, nextY])
                        count -= 1
        return round if count == 0 else -1

    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        """ 207.课程表 """
        ## Again
        from collections import defaultdict
        
        res = 0
        indegree = [0 for _ in range(numCourses)]       # 每门课需要多少前置课 indegree[i]表示 课程i 有 indegree[i] 门前置课
        pre2cur = defaultdict(set)                      # 前置课id:[后置课id]
        for cur, pre in prerequisites:
            indegree[cur] += 1
            pre2cur[pre].add(cur)
        
        queue = [i for i, degree in enumerate(indegree) if degree == 0]     # 先学不需要前置课的
        while queue:
            cur = queue.pop(0)
            res += 1
            for _cur in pre2cur[cur]:
                if indegree[_cur] > 0:
                    indegree[_cur] -= 1
                if indegree[_cur] == 0:
                    queue.append(_cur)
        return res == numCourses


        ## 太绕, 但已经是标准解法
        # from collections import defaultdict

        # res = 0
        # inCounts = [0 for _ in range(numCourses)]      # 初始化 每门课需要多少门前置课程, inCounts[i]表示: 课程i 需要 inCounts[i] 门前置课程
        # pre2cur = defaultdict(set)                          # 初始化 key前置课程:val后置课程
        # for cur, pre in prerequisites:
        #     pre2cur[pre].add(cur)
        #     inCounts[cur] += 1
        # queue = [i for i, count in enumerate(inCounts) if count == 0]       # 不需要前置课的课程
        # while queue:
        #     cur = queue.pop(0)
        #     res += 1
        #     for i in pre2cur[cur]:
        #         if inCounts[i] > 0:
        #             inCounts[i] -= 1
        #         if inCounts[i] == 0:
        #             queue.append(i)
        # return res == numCourses

    def permute(self, nums: List[int]) -> List[List[int]]:
        """ 46.全排列 """
        ## 回溯/递归
        res = []
        path = []
        used = [False] * len(nums)

        def backtrack():
            if len(path) == len(nums):
                res.append(path[:])
                return
            for i in range(len(nums)):
                if used[i]:
                    continue
                used[i] = True
                path.append(nums[i])
                backtrack()
                path.pop()
                used[i] = False
        
        backtrack()
        return res
    
    def subsets(self, nums: List[int]) -> List[List[int]]:
        """ 78.子集 """
        ## 回溯/递归
        res = []
        path = []

        def backtrack(startInd):
            res.append(path[:])
            if startInd == len(nums):
                return
            for i in range(startInd, len(nums)):
                path.append(nums[i])
                backtrack(i + 1)
                path.pop()
        
        backtrack(0)
        return res
    
    def letterCombinations(self, digits: str) -> List[str]:
        """ 17.电话号码的字母组合 \n
            求一个集合的组合, 使用startInd \n
            求多个集合的组合, 不使用startInd, 使用ind指示不同集合 """
        res = []
        path = []
        d2l = {str(k):v for k, v in zip(list(range(2, 10)), 
                                 ['abc', 'def', 'ghi', 'jkl', 'mno', 'pqrs', 'tuv', 'wxyz'])}
        
        def backtrack(ind):
            if ind == len(digits):
                res.append("".join(path[:]))
                return
            letters = d2l[digits[ind]]
            for c in letters:
                path.append(c)
                backtrack(ind + 1)
                path.pop()
        
        backtrack(0)
        return res
    
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        """ 39.组合总和 """
        ## 一个集合 求 组合, 使用startInd. 元素可重复, 回溯调用时i
        res = []
        path = []

        def backtrack(startInd, leftVal):
            if sum(path) == target:
                res.append(path[:])
                return
            elif sum(path) > target:
                return
            for i in range(startInd, len(candidates)):
                path.append(candidates[i])
                backtrack(i, leftVal - candidates[i])
                path.pop()
        
        backtrack(0, target)
        return res
    
    def generateParenthesis(self, n: int) -> List[str]:
        """ 22.括号生成 """
        ## 回溯法, GPT
        res = []
        path = ''

        def backtrack(left, right):
            """
            left:  还能放多少个左括号 \n
            right: 还能放多少个右括号
            """
            nonlocal path
            if left == 0 and right == 0:
                res.append(path)
                return
            if left:
                path += '('
                backtrack(left - 1, right)
                path = path[:-1]
            if right > left:        # 任何时刻, 右括号的数量 < 左括号的数量. 若写为 if right == left, 前面还没有可匹配的左括号; if right, 可能会先放右括号, 就错了
                path += ')'
                backtrack(left, right - 1)
                path = path[:-1]
        
        backtrack(n, n)
        return res
    
    def exist(self, board: List[List[str]], word: str) -> bool:
        """ 79.单词搜索 """
        def func(i, j, word):
            if len(word) == 0:
                return True
            for dir in [-1, 0], [1, 0], [0, 1], [0, -1]:
                nextX = i + dir[0]
                nextY = j + dir[1]
                if nextX < 0 or nextX >= m or nextY < 0 or nextY >= n:
                    continue
                if board[nextX][nextY] == word[0] and not visited[nextX][nextY]:
                    visited[nextX][nextY] = True
                    if func(nextX, nextY, word[1:]):
                        return True
                    visited[nextX][nextY] = False
            return False

        m, n = len(board), len(board[0])
        visited = [[False] * n for _ in range(m)]

        for i in range(m):
            for j in range(n):
                if board[i][j] == word[0] and not visited[i][j]:
                    visited[i][j] = True
                    if func(i, j, word[1:]):
                        return True
                    visited[i][j] = False
        return False
    
    def partition(self, s: str) -> List[List[str]]:
        """ 131.分割回文串 """
        ## 分割--组合--startInd
        res = []
        path = []

        def isValid(s):
            l, r = 0, len(s) - 1
            while l <= r:
                if s[l] != s[r]:
                    return False
                l += 1
                r -= 1
            return True

        def backtrack(startInd):
            if startInd == len(s):
                res.append(path)
                return
            for i in range(startInd, len(s)):
                if isValid(s[startInd:i + 1]):
                    path.append(s[startInd:i + 1])
                    backtrack(i + 1)
                    path.pop()
        
        backtrack(0)
        return res
    
    def searchInsert(self, nums: List[int], target: int) -> int:
        """ 35.搜索插入位置 """
        ## 二分查找 时间:O(logn)
        l, r = 0, len(nums) - 1
        while l <= r:
            m = (l + r) // 2
            if nums[m] == target:
                return m
            elif nums[m] < target:
                l = m + 1
            elif nums[m] > target:
                r = m - 1
        return l
    
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        """ 74.搜索二维矩阵 """
        ## 自己想的, 比较容易想到
        # m, n = len(matrix), len(matrix[0])
        # for i in range(m):
        #     if matrix[i][0] > target:
        #         if i == 0:
        #             return False
        #         else:
        #             i -= 1
        #         break
        # for j in range(n):
        #     if matrix[i][j] == target:
        #         return True
        # return False

        ## 二分查找, 加深理解. 参考 74.搜索二维矩阵
        ## 本题二分查找, 查找第一列时, 找的是第一个<=target的, 所以返回r
        def binSearch(nums, target):
            l, r = 0, len(nums) - 1
            while l <= r:
                m = (l + r) // 2
                if nums[m] == target:
                    return m
                elif nums[m] < target:
                    l = m + 1
                elif nums[m] > target:
                    r = m - 1
            return r

        m, n = len(matrix), len(matrix[0])
        rowInd = binSearch([ln[0] for ln in matrix], target)
        colInd = binSearch(matrix[rowInd], target)
        return matrix[rowInd][colInd] == target
    
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        """ 34.在排序数组中查找元素的第一个和最后一个位置 """
        ## 不要想太复杂: 二分查找 + 中心探测, 至少二分查找会确定是否有target
        if len(nums) == 0:
            return [-1, -1]
        l, r = 0, len(nums) - 1
        while l <= r:
            m = (l + r) // 2
            if nums[m] == target:
                break
            elif nums[m] < target:
                l = m + 1
            elif nums[m] > target:
                r = m - 1
        if nums[m] != target:
            return [-1, -1]
        l = r = m
        while l - 1 >= 0 and nums[l - 1] == nums[m]:
            l -= 1
        while r + 1 <= len(nums) - 1 and nums[r + 1] == nums[m]:
            r += 1
        return [l, r]

    def search(self, nums: List[int], target: int) -> int:
        """ 33.搜索螺旋排序数组 """
        ## 这题有点绕的意思, 没有太懂
        l, r = 0, len(nums) - 1
        while l <= r:
            m = (l + r) // 2
            if nums[m] == target:
                return m
            elif nums[l] < nums[m]:     # 假设某一边有序
                if nums[l] <= target < nums[m]:
                    r = m - 1
                else:
                    l = m + 1
            else:                       # 只能写成else, 写别的就错, 没有太明白
                if nums[m] < target <= nums[r]:
                    l = m + 1
                else:
                    r = m - 1
        return -1
    
    def findMin(self, nums: List[int]) -> int:
        """ 153.寻找旋转排序数组中的最小值 """
        # l, r = 0, len(nums) - 1
        # while l < r:        # while l <= r, 这是经典的普通二分查找. while l < r, 这是区间收缩直到只剩一个元素的写法, 适合本题的 寻找最左边的最小值
        #     m = (l + r) // 2
        #     if nums[m] < nums[-1]:      # m指定落在右边递增那段上了, 所以收缩; m不是落在右半段了吗, m可能就是最小值(最小值只可能在右半段上), 所以让r=m
        #         r = m
        #     else:
        #         l = m + 1               # 因为最小值只可能在右半段, 所以当m落在左半段时可以跳过m
        # return nums[l]                  # while退出就是l==r, 所以最终结果l r均可

        ## Again上面解法
        l, r = 0, len(nums) - 1
        while l < r:
            m = (l + r) // 2
            if nums[m] < nums[-1]:      # m落在右半段
                r = m
            else:
                l = m + 1
        return nums[r]

        ## 错误! 虽然看了官方答案, 自己实现仍然错误!
        # l, r = 0, len(nums) - 1
        # while l <= r:
        #     m = (l + r) // 2
        #     if nums[m] < nums[-1]:
        #         r = m - 1
        #     else:
        #         l = m + 1
        # return l

    def isValid(self, s: str) -> bool:
        """ 20.有效的括号 """
        ## 参考之前做的
        left2right = dict(zip('({[', ')}]'))
        stack = []      # 暂存左括号
        for c in s:
            if c in left2right:
                stack.append(c)
            elif stack and c == left2right[stack[-1]]:
                stack.pop()
            else:
                return False
        return True if not stack else False

        ## 自己想的复杂
        # if len(s) == 1:
        #     return False
        # stack = []
        # left = ['{', '(', '[']
        # right = ['}', ')', ']']
        # for i in range(len(s)):
        #     if s[i] in left:
        #         stack.append(s[i])
        #     else:
        #         if not stack:
        #             return False
        #         _l = stack.pop()
        #         if left.index(_l) != right.index(s[i]):
        #             return False
        # return True if not stack else False
            
    def decodeString(self, s: str) -> str:
        """ 394.字符串解码 """
        res = ''        # 最终的结果
        multi = 0       # 重复次数
        stack = []
        for c in s:
            if c.isdigit():
                multi = multi * 10 + int(c)
            elif c == '[':
                stack.append([multi, res])
                multi = 0
                res = ''
            elif c == ']':
                _multi, _res = stack.pop()
                res = _res + _multi * res
            else:
                res += c
        return res
    
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        """ 739.每日温度 """
        res = [0] * len(temperatures)   # 结果
        stack = []                      # 递减栈
        for i, t in enumerate(temperatures):
            while stack and t > temperatures[stack[-1]]:
                ind = stack.pop()
                res[ind] = i - ind
            stack.append(i)             # 放到这, 维护的依然是递减栈
        return res
    
    def findKthLargest(self, nums: List[int], k: int) -> int:
        """ 215.数组中的第K个最大元素 """
        def quickselect(l, r, k):
            """ 找第k个最小的 """
            pivot = nums[l]
            i, j = l, r
            while i <= j:
                while nums[i] < pivot:
                    i += 1
                while pivot < nums[j]:
                    j -= 1
                if i <= j:
                    nums[i], nums[j] = nums[j], nums[i]
                    i += 1
                    j -= 1
            if k <= j:
                return quickselect(l, j, k)
            elif k >= i:
                return quickselect(i, r, k)
            return nums[k]
        
        n = len(nums)
        return quickselect(0, n - 1, n - k)     # 找倒数第k个, 即正数第n-k个
    
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        """ 347.前K个高频元素 """
        from collections import defaultdict

        d2c = defaultdict(int)
        for n in nums:
            d2c[n] += 1
        d2c = dict(sorted(d2c.items(), key=lambda x: x[1], reverse=True))
        return list(d2c.keys())[:k]
    
    def maxProfit(self, prices: List[int]) -> int:
        """ 121.买卖股票的最佳时机 """
        ## 状态0: 不持有 
        ## 状态1: 持有
        n = len(prices)
        dp = [[0, 0] for _ in range(n)]
        dp[0] = [0, -prices[0]]
        for i in range(1, n):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
            dp[i][1] = max(-prices[i], dp[i - 1][1])
        return dp[-1][0]
    
    def canJump(self, nums: List[int]) -> bool:
        """ 55.跳跃游戏 """
        ## 贪心
        maxReach = 0
        for i, n in enumerate(nums):
            if maxReach < i:
                return False
            maxReach = max(maxReach, i + n)
        return True
    
    def jump(self, nums: List[int]) -> int:
        """ 45.跳跃游戏II """
        res = 0
        start = 0
        end = 1     # 不包含
        maxRange = 0
        while end < len(nums):
            for i in range(start, end):
                maxRange = max(maxRange, i + nums[i])
            
            start = end
            end = maxRange + 1      # 不包含
            res += 1
        return res
    
    def partitionLabels(self, s: str) -> List[int]:
        """ 763.划分字母区间 """
        last = [0] * 26
        for i, c in enumerate(s):
            last[ord(c) - ord('a')] = i
        
        res = []
        start = end = 0
        for i, c in enumerate(s):
            end = max(end, last[ord(c) - ord('a')])
            if i == end:
                res.append(end - start + 1)
                start = end + 1
        return res
    
    def climbStairs(self, n: int) -> int:
        """ 70.爬楼梯 """
        # if n <= 2:
        #     return n
        # dp = [0] * (n + 1)
        # dp[1] = 1
        # dp[2] = 2
        # for i in range(3, n + 1):
        #     dp[i] = dp[i - 1] + dp[i - 2]
        # return dp[-1]

        ## 优化空间
        if n <= 2:
            return n
        dp = [0] * 3
        dp[1] = 1
        dp[2] = 2
        for i in range(3, n + 1):
            tmp = dp[1] + dp[2]
            dp[1], dp[2] = dp[2], tmp
        return dp[-1]
    
    def generate(self, numRows: int) -> List[List[int]]:
        """ 118.杨辉三角 """
        res = []
        for i in range(numRows):
            row = []
            for j in range(i + 1):
                if j == 0 or j == i:
                    row.append(1)
                else:
                    row.append(res[i - 1][j] + res[i - 1][j - 1])
            res.append(row)
        return res
    
    def rob(self, nums: List[int]) -> int:
        """ 198.打家劫舍 """
        n = len(nums)
        if n <= 2:
            return max(nums)
        dp = [0] * n
        dp[0] = nums[0]
        dp[1] = max(nums[:2])
        for i in range(2, n):
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
        return dp[-1]
    
    def numSquares(self, n: int) -> int:
        """ 279.完全平方数 """
        ## 完全背包
        dp = [float('inf')] * (n + 1)
        dp[0] = 0       # 纯为递推
        # 求组合, 不要求有序
        for i in range(1, n + 1):           # 外层for背包
            for j in range(1, int(math.sqrt(i)) + 1):  # 内层for物品
                dp[i] = min(dp[i], dp[i - j ** 2] + 1)
        return dp[-1]
    
    def coinChange(self, coins: List[int], amount: int) -> int:
        """ 322.零钱兑换 """
        ## 完全背包
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        for i in range(len(coins)):
            for j in range(coins[i], amount + 1):
                dp[j] = min(dp[j], dp[j - coins[i]] + 1)
        return dp[-1] if dp[-1] != float('inf') else -1
    
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        """ 139.单词拆分 """
        ## 完全背包. 求排列
        # n = len(s)
        # dp = [False] * (n + 1)      # dp[i] 前i个字符是否能被字典表示
        # dp[0] = True
        # for j in range(1, n + 1):
        #     for i in range(j):
        #         if dp[i] and s[i:j] in wordDict:    # 前i个字符能被表示(索引到i-1) 且 s[i:j]在字典中 --> 总的意思是, 前j个字符(索引到j-1)能否被字典表示
        #             dp[j] = True
        # return dp[-1]

        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True
        for j in range(1, n + 1):
            for word in wordDict:
                if len(word) > j:
                    continue
                if dp[j - len(word)] and s[j - len(word):j] == word:
                    dp[j] = True
        return dp[-1]
    
    def lengthOfLIS(self, nums: List[int]) -> int:
        """ 300.最长递增子序列 """
        n = len(nums)
        dp = [1] * n
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return dp[-1]
    
    def maxProduct(self, nums: List[int]) -> int:
        """ 152.乘积最大子数组 """
        ## 要同时维护最大值和最小值
        n = len(nums)
        dp = [[0, 0] for _ in range(n)]
        dp[0] = [nums[0], nums[0]]
        for i in range(1, n):
            if nums[i] >= 0:
                dp[i][0] = min(nums[i], nums[i] * dp[i - 1][0])
                dp[i][1] = max(nums[i], nums[i] * dp[i - 1][1])
            else:
                dp[i][0] = min(nums[i], nums[i] * dp[i - 1][1])
                dp[i][1] = max(nums[i], nums[i] * dp[i - 1][0])
        return max(max(l) for l in dp)
    
    def canPartition(self, nums: List[int]) -> bool:
        """ 416.分割等和子集 """
        if sum(nums) % 2:
            return False
        target = sum(nums) // 2
        dp = [0] * (target + 1)
        # 一维dp, 外层for物品 内层for背包且背包倒序, 避免覆盖前面的
        for i in range(len(nums)):
            for j in range(target, nums[i] - 1, -1):
                dp[j] = max(dp[j], dp[j - nums[i]] + nums[i])
        return dp[-1] == target
    
    def uniquePaths(self, m: int, n: int) -> int:
        """ 62.不同路径 """
        dp = [[0] * n for _ in range(m)]
        for i in range(m):
            dp[i][0] = 1
        for j in range(n):
            dp[0][j] = 1
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[-1][-1]
    
    def minPathSum(self, grid: List[List[int]]) -> int:
        """ 64.最小路径和 """
        m, n = len(grid), len(grid[0])
        for i in range(m):
            for j in range(n):
                if i == 0 and j >= 1:       # 第0行
                    grid[i][j] += grid[i][j - 1]
                elif j == 0 and i >= 1:     # 第0列
                    grid[i][j] += grid[i - 1][j]
                elif i >= 1 and j >= 1:
                    grid[i][j] += min(grid[i - 1][j], grid[i][j - 1])
        return grid[-1][-1]
    
    def longestPalindrome(self, s: str) -> str:
        """ 5.最长回文子串 """
        ## 两边探测
        res = ''
        n = len(s)
        for i in range(n):
            l, r = i - 1, i + 1
            while l >= 0 and s[l] == s[i]:
                l -= 1
            while r <= n - 1 and s[r] == s[i]:
                r += 1
            while 0 <= l <= r <= n - 1 and s[l] == s[r]:
                l -= 1
                r += 1
            l += 1
            r -= 1
            res = s[l:r + 1] if r - l + 1 > len(res) else res
        return res
    
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        """ 1143.最长公共子序列 """
        m = len(text1)
        n = len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return max(max(l) for l in dp)
    
    def minDistance(self, word1: str, word2: str) -> int:
        """ 72.编辑距离 """
        m = len(word1)
        n = len(word2)
        dp = [[float('inf')] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            dp[i][0] = i
        for j in range(1, n + 1):
            dp[0][j] = j
        dp[0][0] = 0
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
        return dp[-1][-1]

    def singleNumber(self, nums: List[int]) -> int:
        """ 136.只出现一次的数字 """
        # nums.sort()

        # stack = []
        # for n in nums:
        #     if stack and n == stack[-1]:
        #         stack.pop()
        #     else:
        #         stack.append(n)
        # return stack[-1]

        ## 异或
        res = nums.pop(0)
        while nums:
            res ^= nums.pop(0)
        return res
    
    def majorityElement(self, nums: List[int]) -> int:
        """ 169.多数元素 """
        counts = 0
        res = 0
        for n in nums:
            if counts == 0:
                res = n
            counts += (1 if n == res else -1)
        return res

    def sortColors(self, nums: List[int]) -> None:
            """
            75.颜色分类 \n
            经典荷兰国旗问题 \n
            Do not return anything, modify nums in-place instead.
            """
            ## 单指针
            # n = len(nums)
            # ptr = 0     # 索引[0, ptr - 1]是头部, 用户交换元素
            # for i in range(n):
            #     if nums[i] == 0:
            #         nums[i], nums[ptr] = nums[ptr], nums[i]
            #         ptr += 1
            # for i in range(ptr, n):
            #     if nums[i] == 1:
            #         nums[i], nums[ptr] = nums[ptr], nums[i]
            #         ptr += 1

            ## 双指针
            n = len(nums)
            pt0 = pt1 = 0
            for i in range(n):
                if nums[i] == 1:
                    nums[i], nums[pt1] = nums[pt1], nums[i]
                    pt1 += 1
                if nums[i] == 0:
                    nums[i], nums[pt0] = nums[pt0], nums[i]
                    if pt0 < pt1:
                        nums[i], nums[pt1] = nums[pt1], nums[i]
                    pt0 += 1
                    pt1 += 1        # 0的部分 1的部分, 你想, 前面插入0, pt1自然要向后移

    def nextPermutation(self, nums: List[int]) -> None:
            """
            31.下一个排列 \n
            Do not return anything, modify nums in-place instead.
            """
            if len(nums) == 1:
                return
            i, j = len(nums) - 2, len(nums) - 1
            # 从右往左, 找第一个升序的相邻数对
            while i < j and i >= 0:
                if nums[i] < nums[j]:       # 对nums[i]下手
                    break
                i -= 1
                j -= 1
            # 对nums[i]下手
            if i >= 0:
                k = len(nums) - 1
                while i < k:
                    if nums[i] < nums[k]:
                        break
                    k -= 1
                if i < k:
                    nums[i], nums[k] = nums[k], nums[i]
                    nums[i + 1:] = sorted(nums[i + 1:])
            else:
                nums.sort()

    def findDuplicate(self, nums: List[int]) -> int:
        """ 287.寻找重复数 """
        ## 思路: 环形链表 的 快慢指针
        # 初始化快慢指针
        slow = 0
        fast = 0
        # 指针走一步
        slow = nums[slow]
        fast = nums[nums[fast]]
        while slow != fast:
            slow = nums[slow]
            fast = nums[nums[fast]]
        # while退出到达相遇点. 分别从起点 相遇点出发, 一步一步走肯定会在环入口相遇
        start0 = 0
        start1 = slow
        while start0 != start1:
            start0 = nums[start0]
            start1 = nums[start1]
        return start0

if __name__ == '__main__':
    sl = Solution()

    s = "leetcode"
    wordDict = ["leet", "code"]
    print(sl.wordBreak(s, wordDict))