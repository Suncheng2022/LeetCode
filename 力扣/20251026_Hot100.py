'''
10.26   执行力这么差?
'''
from typing import List, Optional

# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

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
        ## 动态规划
        dp = [0] * len(nums)        # dp[i] 以nums[i]结尾的连续子数组和为dp[i]
        dp[0] = nums[0]
        for i in range(1, len(nums)):
            dp[i] = max(nums[i], dp[i - 1] + nums[i])
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

if __name__ == '__main__':
    sl = Solution()

    matrix = [[1,2,3],[4,5,6],[7,8,9]]
    print(sl.rotate(matrix))
    print(matrix)