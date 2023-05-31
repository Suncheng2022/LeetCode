from typing import List, Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        """
        2.两数相加
        2023.05.23 中等
        题解：用最快的方法解决
        """
        num1, num2 = 0, 0
        stack1, stack2 = [], []
        while l1:
            stack1.append(l1.val)
            l1 = l1.next
        while l2:
            stack2.append(l2.val)
            l2 = l2.next
        # stack1.reverse()
        # stack2.reverse()
        print(stack1, stack2)
        while stack1:
            num1 *= 10
            num1 += int(stack1.pop())
        while stack2:
            num2 *= 10
            num2 += int(stack2.pop())
        num = num1 + num2
        tmp = head = ListNode()
        if not num:
            return ListNode()
        while num:
            tmp.next = ListNode(num % 10)
            num //= 10
            tmp = tmp.next

        return head.next

    def lengthOfLongestSubstring(self, s: str) -> int:
        """
        3.无重复字符的最长字串
        2023.05.23 中等
        题解：j向右探测
        """
        if len(s) < 2:
            return len(s)
        i, j = 0, 1
        lengths = []
        while j < len(s):
            if s[j] in s[i:j]:
                ind = s[i:j].index(s[j]) + i
                i = ind + 1
            lengths.append(j - i + 1)
        return max(lengths)

    def longestPalindrome(self, s: str) -> str:
        """
        5.最长回文字串
        2023.05.23 中等
        题解：中心扩散法。对每个元素，向两边探测
        """
        max_len = 0
        for i in range(len(s)):
            cur_len = 1
            left, right = i - 1, i + 1
            while left >= 0 and s[left] == s[i]:
                cur_len += 1
                left -= 1
            while right < len(s) and s[right] == s[i]:
                cur_len += 1
                right += 1
            while left >= 0 and right < len(s) and s[left] == s[right]:
                cur_len += 2
                left -= 1
                right += 1
            if cur_len > max_len:
                max_len = cur_len
                startInd = left + 1
        return s[startInd:startInd+max_len]

    def maxArea(self, height: List[int]) -> int:
        """
        11.盛水最多的容器
        2023.05.23 中等
        题解：移动长板不会使新的短板变长，所以移动短板
        """
        i, j = 0, len(height) - 1
        maxArea = 0     # 记录最大面积/容积
        while i < j:
            curArea = (j - i) * min(height[j], height[i])
            maxArea = curArea if curArea > maxArea else maxArea
            if height[i] < height[j]:
                i += 1
            else:
                j -= 1
        return maxArea

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        """
        15.三数之和
        2023.05.23 中等
        题解：1.升序 2.对每一个i，寻找left、right
        """
        nums.sort()
        res = []
        for i in range(len(nums) - 2):
            if nums[i] > 0:     # 因为升序排序了，所以若nums[i]大于0则后面肯定找不到left、right，也不用再找了
                break
            left, right = i + 1, len(nums) - 1  # 对每一个i，找所有符合的left、right
            while left < right:
                if nums[i] + nums[left] + nums[right] == 0:
                    if [nums[i], nums[left], nums[right]] not in res:
                        res.append([nums[i], nums[left], nums[right]])
                    left += 1
                elif nums[i] + nums[left] + nums[right] < 0:
                    left += 1
                else:
                    right -= 1
        return res

    def letterCombinations(self, digits: str) -> List[str]:
        """
        17.电话号码的字母组合
        2023.05.24 中等
        题解：解法一 暴力
        """
        dig2let = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}
        if len(digits) == 0:
            return []
        elif len(digits) == 1:
            return list(dig2let[digits])
        res = ['']      # 不能为空，否则下面遍历得不到结果；也可以用第一个数字对应的字母列表初始化
        for dig in digits:
            tmp = []
            for let in list(dig2let[dig]):
                tmp.extend([item + let for item in res])
            res = tmp
        return res

    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        """
        19.删除链表的倒数第N个结点
        2023.05.24 中等
        题解：把所有节点地址放到数组中，用索引[-n]取倒数第n个节点——自己做竟然对了，比之前的解法更好理解
        """
        tmpHead = head
        nodes = []
        while head:
            nodes.append(head)
            head = head.next
        if n == len(nodes):
            return nodes[0].next
        prev = nodes[- n - 1]
        cur = nodes[-n]
        prev.next = cur.next
        return tmpHead

    def isValid(self, s: str) -> bool:
        """
        20.有效的括号
        2023.05.25 简单
        题解：栈呗
        """
        # 单纯使用栈还是比较复杂，边界情况不好考虑
        # myDict = dict(zip(list('()[]{}'), [-1, 1, -2, 2, -3, 3]))
        # stack = []      # 保存数字代替括号
        # s = list(s)
        # while s:
        #     c = s.pop(0)
        #     if len(stack) and myDict[c] + stack[-1] == 0 and stack[-1] < 0:
        #         stack.pop()
        #     else:
        #         stack.append(myDict[c])
        # return True if not len(stack) else False

        # 方法二 效率比较高 也比较好想
        myDict = dict(zip(list("([{"), list(")]}")))
        stack = []
        for c in s:
            if c in myDict:     # 如果是左括号，直接入栈
                stack.append(c)
            elif len(stack) and c == myDict[stack[-1]]:     # 如果遇到右括号，则检查是否能抵消，不能抵消就结束——肯定不匹配了
                stack.pop()
            else:
                return False
        return True if not len(stack) else False

    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        """
        21.合并两个有序链表
        2023.05.27 简单
        """
        cur = head = ListNode()
        while list1 and list2:
            if list1.val < list2.val:
                cur.next = list1
                list1 = list1.next
            else:
                cur.next = list2
                list2 = list2.next
            cur = cur.next
        if list1:
            cur.next = list1
        else:
            cur.next = list2
        return head.next

    def generateParenthesis(self, n: int) -> List[str]:
        """
        22.括号生成
        2023.05.27
        题解：'(' + p组括号 + ')' + q组括号，p + q = n - 1
        """
        total_l = [[None], ['()']]      # 0组、1组括号的情况
        for i in range(2, n + 1):   # 依次处理2组~n组
            l = []  # i组时的组合情况
            for j in range(i):      # 遍历p、q，p+q=i-1得到i组的组合情况
                p = total_l[j]
                q = total_l[i - 1 - j]
                for k1 in p:
                    for k2 in q:
                        if k1 == None:
                            k1 = ''
                        if k2 == None:
                            k2 = ''
                        l.append('(' + k1 + ')' + k2)
            total_l.append(l)
        return total_l[-1]

    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        31.下一个排列
        2023.05.27 中等
        题解：从右向左定位到第一个升序对，将较小的数与后面次小的数交换，再将后面的升序，完成
        """
        # 特别处理只有1、2个元素的情况
        if len(nums) == 1:
            return
        elif len(nums) == 2:
            nums.reverse()
            return
        # 从右向左寻找第一对升序对
        i, j = len(nums) - 2, len(nums) - 1
        while nums[i] >= nums[j]:   # 条件是降序，循环结束的时候就找到了第一对升序
            # 都从右比较到最左边了，还是没找到升序，即整个序列是非升序/降序，此时已经是最大了，升序返回最小的数就行了
            if i - 1 < 0:
                nums.sort()
                return
            i -= 1
            j -= 1
        # 找到第一对升序对索引[i, j]后，将i与后面次小的数交换，然后升序
        # 之所以能定位到索引[i, j]为升序，是因为其后面的都为【非升序】。所以从右能找到比索引i大的次小数
        for k in range(len(nums) - 1, j - 1, -1):
            if nums[k] > nums[i]:   # 找到比索引i大的次小数
                nums[k], nums[i] = nums[i], nums[k]
                nums[j:] = sorted(nums[j:])     # 这里要注意，使用nums[j:].sort()得不到正确结果
                break

    def search(self, nums: List[int], target: int) -> int:
        """
        33.搜索旋转排序数组
        2023.05.30 中等
        题解：O(logn)且有序，想到二分查找。数组从中间分开肯定是有一半有序的，判断哪一半有序，从中找是否有target
        """
        l, r = 0, len(nums) - 1
        while l <= r:
            m = (l + r) // 2
            if nums[m] == target:
                return m
            if nums[l] <= nums[m]:  # 假如索引1~m是升序
                if nums[l] <= target < nums[m]:
                    r = m - 1
                else:
                    l = m + 1
            else:   # 假如另一半是升序
                if nums[m] < target <= nums[r]:
                    l = m + 1
                else:
                    r = m - 1
        return -1

    def searchRange(self, nums: List[int], target: int) -> List[int]:
        """
        34.在排序数组中查找元素的第一个和最后一个位置
        2023.05.30 中等
        题解：题目要求O(logn)且又给了升序的条件，想到二分查找，又想到33.
        """
        l, r = 0, len(nums) - 1
        while l <= r:
            m = (l + r) // 2
            if nums[m] == target:
                left = m - 1
                while left >= 0 and nums[left] == target:
                    left -= 1
                left += 1   # 这个后处理，因为while跳出的时候，要么越界、要么不等，所以left要往回走一下

                right = m + 1
                while right <= len(nums) - 1 and nums[right] == target:
                    right += 1
                right -= 1
                return [left, right]
            elif nums[m] < target:
                l = m + 1
            else:
                r = m - 1
        return [-1, -1]


    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        73.矩阵置零
        2023.05.30 中等
        题解：面试题 遍历记录0元素行列索引，再遍历置零
        """
        numRows, numCols = len(matrix), len(matrix[0])
        zeroInds = []
        for i in range(numRows):
            for j in range(numCols):
                if matrix[i][j] == 0:
                    zeroInds.append([i, j])
        for i, j in zeroInds:
            matrix[i] = [0] * numCols
            for k in range(numRows):
                matrix[k][j] = 0


    def numIslands(self, grid: List[List[str]]) -> int:
        """
        200.岛屿数量
        2023.05.17 中等
        题解：dfs
        :param grid:
        :return:
        """
        def dfs(grid, x, y):
            grid[x][y] = 0
            nr = len(grid)
            nc = len(grid[0])
            for x, y in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                if 0 <= x < nr and 0 <= y < nc and grid[x][y] == "1":
                    dfs(grid, x, y)

        nr = len(grid)
        if nr == 0:
            return 0
        nc = len(grid[0])
        num_islands = 0
        for x in range(nr):
            for y in range(nc):
                if grid[x][y] == "1":
                    num_islands += 1
                    dfs(grid, x, y)
        return num_islands

    def rob(self, nums: List[int]) -> int:
        """
        198.打家劫舍
        2023.05.22 中等
        题解：直接看之前代码就能瞬间理解
        """
        n = len(nums)
        dp = [-1] * (n + 1)
        dp[0] = 0
        dp[1] = nums[0]
        for k in range(2, n + 1):
            dp[k] = max(dp[k - 1], dp[k - 2] + nums[k - 1])
        return dp[n]

    def majorityElement(self, nums: List[int]) -> int:
        """
        169.多数元素
        2023.05.22 简单
        """
        nums.sort()
        return nums[len(nums)//2]


if __name__ == '__main__':
    sl = Solution()

    nums = [1, 3, 2]
    sl.nextPermutation(nums)
    print(nums)

    # print(sl.generateParenthesis(3))
    # s = "(){}}{"
    # print(sl.isValid(s))

    # digits = "23"
    # print(sl.letterCombinations(digits))

    # l1 = [2, 4, 3]
    # l2 = [5, 6, 4]
    # print(sl.addTwoNumbers(l1, l2))