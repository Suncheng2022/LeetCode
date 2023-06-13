import copy
import random
from typing import List, Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class MinStack:
    """
    155.最小栈
    2023.06.12 中等
    题解：就是实现一个栈，唯一注意的是跟踪最小元素
    """
    # 答案，维护最小值的方法很巧妙！
    def __init__(self):
        # 注意维护最小值的处理方法，很巧妙
        self.stack = []
        self.minVal = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        if self.minVal.__len__() == 0:
            self.minVal.append(val)
        else:
            self.minVal.append(min(val, self.minVal[-1]))       # 【维护最小值，巧妙！】

    def pop(self) -> None:
        _ = self.stack.pop()
        _ = self.minVal.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.minVal[-1]

    # 自己写的这个，不太好，思路比较模糊，没创意
    # def __init__(self):
    #     self.stack = []
    #     self.minVal = float('inf')
    #
    # def push(self, val: int) -> None:
    #     # 插入时维护最小值
    #     self.stack.append(val)
    #     self.minVal = min(self.stack)
    #
    # def pop(self) -> None:
    #     # 弹出时要维护最小值
    #     self.stack.pop()
    #     if self.stack.__len__():
    #         self.minVal = min(self.stack)
    #
    # def top(self) -> int:
    #     return self.stack[-1]
    #
    # def getMin(self) -> int:
    #     return self.minVal

class LRUCache:
    """
    146.LRU缓存
    2023.06.11 中等
    题解：自己实现 and OrderedDict，另一个方法不好记
    """
    # 自己实现的方法，复杂度高
    # def __init__(self, capacity: int):
    #     self.capacity = capacity
    #     self.keys = []       # 维护最近使用状态
    #     self.cache = {}
    #
    # def get(self, key: int) -> int:
    #     if key in self.keys:
    #         self.keys.remove(key)
    #         self.keys.insert(0, key)
    #         return self.cache[key]
    #     return -1
    #
    # def put(self, key: int, value: int) -> None:
    #     if key in self.keys:
    #         self.keys.remove(key)
    #         self.keys.insert(0, key)
    #         self.cache[key] = value
    #     else:
    #         self.keys.insert(0, key)
    #         self.cache[key] = value
    #         if len(self.keys) > self.capacity:
    #             k = self.keys.pop()
    #             self.cache.pop(k)

    # 答案之不推荐之我只能记住这个了，OrderedDict
    def __init__(self, capacity: int):
        from collections import OrderedDict

        self.capacity = capacity
        self.d = OrderedDict()

    def get(self, key: int) -> int:
        if key in self.d:
            self.d.move_to_end(key, last=False)
            return self.d[key]
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.d:
            self.d.move_to_end(key, last=False)
            self.d[key] = value
        else:
            self.d[key] = value
            self.d.move_to_end(key, last=False)
            if self.d.__len__() > self.capacity:
                self.d.popitem(last=True)


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

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        """
        39.组合总和
        2023.06.01 中等
        题解：回溯[深度优先]
        """
        def dfs(candidates, path, res, begin, size, target):
            if target < 0:
                return
            elif target == 0:
                res.append(path)
                return
            else:
                for i in range(begin, size):
                    dfs(candidates, path + [candidates[i]], res, i, size, target - candidates[i])

        size = len(candidates)
        if size == 0:
            return []
        path = []
        res = []
        dfs(candidates, path, res, 0, size, target)
        return res

    def permute(self, nums: List[int]) -> List[List[int]]:
        """
        46.全排列
        2023.06.01
        题解：回溯[深度优先]
        """
        def dfs(nums, path, res, depth, size, used):
            if depth == size:
                res.append(path[:])
                return
            for i in range(size):
                if not used[i]:
                    path.append(nums[i])
                    used[i] = True
                    dfs(nums, path, res, depth + 1, size, used)
                    path.pop()
                    used[i] = False

        size = len(nums)
        if size == 0:
            return []
        path, res = [], []
        used = [False for _ in range(size)]
        dfs(nums, path, res, 0, size, used)
        return res

    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        48.旋转图像
        2023.06.02 中等
        题解：原矩阵第row行第i个元素 对应 旋转矩阵倒数第row列的第个元素
        """
        n = len(matrix)
        matrix_res = copy.deepcopy(matrix)
        for i in range(n):
            for j in range(n):
                # 我知道错的原因了，这里不是两者交换，是要每一个元素值分别映射到正确的位置
                # matrix[i][j], matrix[j][-i - 1] = matrix[j][-i - 1], matrix[i][j]
                matrix_res[j][-i - 1] = matrix[i][j]
        # print(matrix_res)
        matrix[:] = matrix_res      # 这样给原来的矩阵赋值

    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        """
        49.字母异位词分组
        2023.06.03 中等
        题解：首先想到的是用set
        """
        # 参考答案，一次遍历就能完成
        from collections import defaultdict
        res = defaultdict(list)     # 定义一个val为list类型的字典
        for word in strs:
            res["".join(sorted(word))].append(word)
        return list(res.values())

        # 自己写的，复杂度有点高
        # tmp = set()
        # for word in strs:
        #     if "".join(sorted(word)) not in tmp:
        #         tmp.add("".join(sorted(word)))
        # tmp = {key: [] for key in tmp}
        # for word in strs:
        #     tmp["".join(sorted(word))].append(word)
        # return list(tmp.values())

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

    def search(self, nums: List[int], target: int) -> int:
        """
        33.搜索旋转排序数组
        2023.05.28 中等
        题解：O(logn)则二分查找
        """
        l, r = 0, len(nums) - 1
        while l <= r:   # 二分查找，似乎都是≤
            m = (l + r) // 2
            if nums[m] == target:
                return m
            if nums[l] <= nums[m]:  # 判断升序
                if nums[l] <= target < nums[m]:
                    r = m - 1
                else:
                    l = m + 1
            else:   # 判断升序
                if nums[m] < target <= nums[r]:
                    l = m + 1
                else:
                    r = m - 1
        return -1

        # 参考答案写的
        # l, r = 0, len(nums) - 1
        # while l <= r:
        #     m = (l + r) // 2
        #     if nums[m] == target:
        #         return m
        #     if nums[l] <= nums[m]:  # 如果左半段是升序
        #         if nums[l] <= target < nums[m]:
        #             r = m - 1
        #         else:
        #             l = m + 1
        #     else:
        #         if nums[m] < target <= nums[r]:
        #             l = m + 1
        #         else:
        #             r = m - 1
        # return -1

    def searchRange(self, nums: List[int], target: int) -> List[int]:
        """
        34.在排序数组中查找元素的第一个和最后一个位置
        2023.05.28 中等
        题解：参考33. 要求在有序数组实现时间复杂度O(logn)则想到二分查找
        """
        l, r = 0, len(nums) - 1
        pos = -1
        while l <= r:
            m = (l + r) // 2
            if nums[m] == target:
                pos = m
                break
            elif nums[m] < target:
                l = m + 1
            else:
                r = m - 1
        if pos == -1:
            return [-1, -1]
        # 沿索引pos向左探测
        left = pos - 1
        while left >= 0 and nums[left] == nums[pos]:
            left -= 1
        left += 1   # 思考一下while的终止条件，要么越界跳出，要么值不相等跳出。要让left回到最边上且相等的位置上
        # 沿索引pos向右探测
        right = pos + 1
        while right <= len(nums) - 1 and nums[right] == nums[pos]:
            right += 1
        right -= 1
        return [left, right]

    def maxSubArray(self, nums: List[int]) -> int:
        """
        53.最大子数组和
        2023.06.03 中等
        题解：直接看答案代码，一下明了：自己想的思路还是有一部分是对的，比如每个索引位置都计算一个最大值。
                                    遍历，前一个最大值<=0，则当前值加上的话只会不变或变小；
                                         前一个最大值>0，则当前值加上的话只会变大，所以加上。
        """
        if len(nums) == 1:
            return nums[0]
        resList = [nums[0]]
        for i in range(1, len(nums)):
            if resList[-1] <= 0:
                resList.append(nums[i])
            else:
                resList.append(nums[i] + resList[-1])
        return max(resList)

    def canJump(self, nums: List[int]) -> bool:
        """
        55.跳跃游戏
        2023.06.03 中等
        题解：直接看答案吧，一下明了
        """
        maxStep = 0
        for i, num in enumerate(nums):
            if i <= maxStep < i + nums[i]:
                maxStep = i + nums[i]
        return maxStep >= len(nums) - 1

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        """
        56.合并区间
        2023.06.04
        题解：前一个尾元素 <= 后一个的首元素 则合并
        """
        intervals.sort(key=lambda x: x[0])
        res = []
        for inter in intervals:
            if not res or res[-1][-1] < inter[0]:
                res.append(inter)
            else:
                res[-1][-1] = max(res[-1][-1], inter[1])
        return res

    def uniquePaths(self, m: int, n: int) -> int:
        """
        62.不同路径
        2023.06.04 中等
        """
        res = [[1] * n] + [[1] + [0] * (n - 1) for _ in range(m - 1)]
        for i in range(1, m):
            for j in range(1, n):
                res[i][j] = res[i - 1][j] + res[i][j - 1]
        return res[-1][-1]

    def minPathSum(self, grid: List[List[int]]) -> int:
        """
        64.最小路径和
        2023.06.04 中等
        """
        m = len(grid)
        n = len(grid[0])
        # res = [[grid[0][0]] + [0] * (n - 1)] + [[0] * n for _ in range(m - 1)]
        for i in range(0, m):
            for j in range(0, n):
                if i - 1 < 0 and j - 1 < 0:     # 第0行第0列
                    grid[i][j] = grid[i][j]
                elif i - 1 < 0 and j - 1 >= 0:  # 第0行
                    grid[i][j] += grid[i][j - 1]
                elif i - 1 >= 0 and j - 1 < 0:  # 第0列
                    grid[i][j] += grid[i - 1][j]
                elif i - 1 >= 0 and j - 1 >= 0: # 除第0行第0列之外的元素
                    grid[i][j] = min(grid[i][j] + grid[i - 1][j], grid[i][j] + grid[i][j - 1])
        return grid[-1][-1]

    def climbStairs(self, n: int) -> int:
        """
        70.爬楼梯
        2023.06.05 简单
        题解：费锲那波数列
        """
        # dp = [-1] * n
        # if n == 1:
        #     return 1
        # dp[0] = 1
        # dp[1] = 2
        # for i in range(2, n):
        #     dp[i] = dp[i - 1] + dp[i - 2]
        # return dp[-1]

        # 不能处理n=1的情况
        if n == 1:
            return 1
        dp = [-1] * (n + 1)
        dp[1] = 1
        dp[2] = 2
        for i in range(3, len(dp)):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[-1]

    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        75.颜色分类
        2023.06.05 中等
        题解：原地排序，也就是不使用额外空间，O(1)可不可以呢。不就是从小到大排序吗
        """
        # 答案 双指针
        p0 = p1 = 0
        for i in range(len(nums)):
            if nums[i] == 0:
                # 遍历到0，与p0交换；
                # 若此时p0<p1，肯定把1换走了，因为p0、p1初始指向同一位置，需要把1再换回到1元素的末尾；
                # 要注意p0、p1都要往后移动
                nums[i], nums[p0] = nums[p0], nums[i]
                if p0 < p1:
                    nums[i], nums[p1] = nums[p1], nums[i]
                p0 += 1
                p1 += 1
            elif nums[i] == 1:
                nums[i], nums[p1] = nums[p1], nums[i]
                p1 += 1


        # 冒泡试一下
        # for i in range(len(nums) - 1):
        #     for j in range(len(nums) - 1 - i):
        #         if nums[j] > nums[j + 1]:
        #             nums[j], nums[j + 1] = nums[j + 1], nums[j]

    def subsets(self, nums: List[int]) -> List[List[int]]:
        """
        78.子集
        2023.06.05 中等
        题解：能想到递归回溯 https://programmercarl.com/0078.%E5%AD%90%E9%9B%86.html#c-%E4%BB%A3%E7%A0%81
        """
        def backtracking(startInd):
            res.append(path[:])
            if startInd >= len(nums):
                return
            for i in range(startInd, len(nums)):
                path.append(nums[i])
                backtracking(i + 1)
                path.pop()

        path = []
        res = []
        backtracking(0)
        return res

    def exist(self, board: List[List[str]], word: str) -> bool:
        """
        79.单词搜索
        2023.06.06 中等
        题解：似乎也是回溯. 回溯方法中cur_i cur_j变量很重要，否则错误
        """
        # 重写一遍
        def backtracking(i, j, word):
            if len(word) == 0:      # 代表回溯过程中word已经全部匹配到
                return True
            for d in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                cur_i = i + d[0]    # 这里另定义cur_i, cur_j是很重要的，为什么？--每层都定义新的索引，否则会影响上一层的递归
                cur_j = j + d[1]
                if 0 <= cur_i < m and 0 <= cur_j < n and board[cur_i][cur_j] == word[0]:
                    if marked[cur_i][cur_j]:    # 表示这个方向的元素已经探索过了，则探索另一个方向
                        continue
                    marked[cur_i][cur_j] = True
                    if backtracking(cur_i, cur_j, word[1:]):
                        return True
                    marked[cur_i][cur_j] = False
            return False

        m, n = len(board), len(board[0])
        marked = [[False] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if board[i][j] == word[0]:
                    marked[i][j] = True
                    if backtracking(i, j, word[1:]):
                        return True
                    marked[i][j] = False
        return False

        # m = len(board)
        # n = len(board[0])
        # directs = [(-1, 0), (1, 0), (0, -1), (0, 1)]     # 上下左右四个方向
        # marked = [[False] * n for _ in range(m)]
        #
        # def backtracking(i, j, word):
        #     if len(word) == 0:
        #         return True
        #     for d in directs:
        #         cur_i = i + d[0]
        #         cur_j = j + d[1]
        #         if 0 <= cur_i < m and 0 <= cur_j < n and board[cur_i][cur_j] == word[0]:
        #             if marked[cur_i][cur_j]:    # 这个方向探索过，尝试下一个方向
        #                 continue
        #             marked[cur_i][cur_j] = True
        #             if backtracking(cur_i, cur_j, word[1:]):
        #                 return True
        #             else:
        #                 marked[cur_i][cur_j] = False
        #     return False
        #
        # for i in range(m):
        #     for j in range(n):
        #         if board[i][j] == word[0]:
        #             marked[i][j] = True
        #             if backtracking(i, j, word[1:]):
        #                 return True
        #             else:
        #                 marked[i][j] = False    # 回溯
        # return False

    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        """
        94.二叉树的中序遍历
        2023.06.06 简单
        题解：还是看答案吧，并不简单
        """
        # 迭代：其实是模拟递归
        stack = []      # 模拟递归的栈
        res = []
        while stack or root:    # 初次进入stack空，后面不断遍历可能root空
            if root:    # 先通过while循环把所有“左孩子”放进 栈，方便后面的迭代；所有的“左孩子”里是包含“中间孩子”的概念的，细想一下，对吧
                stack.append(root)
                root = root.left
            else:       # “左孩子”都消耗完了，则“右孩子”开始进 栈
                node = stack.pop()
                res.append(node.val)
                root = node.right
        return res

        # 递归。若3个if改为if elif elif结构，是错误的，这样只能判断1个条件了，而不是3个if时能判断3个条件。
        # 比如node是叶子节点时就该添加到res，而不是什么都不做。
        res = []

        def fun(node):
            if node and node.left:  # 当前节点不空，且有左孩子节点
                fun(node.left)
            if node:
                res.append(node.val)
            if node and node.right:
                fun(node.right)

        fun(root)
        return res

    def numTrees(self, n: int) -> int:
        """
        96.不同的二叉搜索树
        2023.06.06 中等
        题解：step1.若k为root，则1~k-1构建左子树、k+1~n构建右子树
             step2.设有i个数构建子树，除去根节点还有i-1个节点，则有j个数构建左子树、i-1-j个数构建右子树；即i个数能构建BST共那么多个
        """
        dp = [0] * (n + 1)
        dp[0] = 1   # 0个节点只能构成1种BST，空树
        dp[1] = 1   # 1个节点只能构成1种BST，只有1个节点的树
        for i in range(2, len(dp)):     # 计算i个数能组成多少种BST
            for j in range(0, i):   # 注意j的范围[0, i-1]，则另一子树范围为对应的i-1-j，因为一共有i-1个节点
                dp[i] += (dp[j] * dp[i - 1 - j])
        return dp[n]

    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        """
        98.验证二叉搜索树
        2023.06.07 中等
        题解：对BST中序遍历，当前访问节点值肯定大于前一节点值，否则不是BST
             递归思路判断BST反而不好实现
        """
        # 迭代 中序遍历
        stack = []
        res = []
        while stack or root:    # root不空，则不断收集“左孩子”；root空，stack不空，弹出节点
            if root:
                stack.append(root)
                root = root.left
            else:
                node = stack.pop()
                if len(res) and node.val <= res[-1]:    # BST的中序遍历严格递增
                    return False
                res.append(node.val)
                root = node.right
        return True

    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        """
        101.对称二叉树
        2023.06.07 简单
        题解：首先想到了依然是中序遍历，自己写递归有测试用例没通过，即考虑的不完全
            答案 递归，但不是中序遍历的思路
        """
        # 递归
        # def dfs(left, right):
        #     if not (left or right):     # 都空
        #         return True
        #     elif not (left and right):  # 仅1个空-->你想呀，上面判断都空，能到这里肯定不是都空，至少有1个；若2个均不空，进不去
        #         return False
        #     elif left.val != right.val: # 都不空，但值不等-->经过上面两个判断，这里只能是都不空
        #         return False
        #     else:                       # 都不空，值相等
        #         return dfs(left.left, right.right) and dfs(left.right, right.left)
        #
        # return dfs(root.left, root.right)

        # 迭代
        # 基本的判断条件还是不变
        if not root or not (root.left or root.right):   # 如果root空 或者 root不空但没有孩子节点
            return True
        queue = [root.left, root.right]  # 用辅助队列实现迭代；放入的时候必须同时放入两个，取的时候必须同时取两个
        while queue:
            # 还是那几种情况的判断，但注意判断之后的执行语句
            left = queue.pop(0)
            right = queue.pop(0)
            if not (left or right):     # 均空
                continue
            elif not (left and right):  # 至少有1个不空（考虑上个判断），若要进入此条件则只能1个空1个不空
                return False
            elif left.val != right.val: # 均不空（考虑上面两个判断），但值不能
                return False
            else:                       # 均不空，且值相等
                queue.append(left.left)
                queue.append(right.right)
                queue.append(left.right)
                queue.append(right.left)
        return True

    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        """
        102.二叉树的层序遍历
        2023.06.07 中等
        题解：这题还中等，简单
        """
        if not root:    # root为空
            return []
        queue = [root]
        res = []
        while queue:
            tmpVals = []
            tmpNodes = []
            while queue:
                node = queue.pop(0)
                if node:
                    tmpVals.append(node.val)
                    tmpNodes.append(node)
            res.append(tmpVals)
            queue = tmpNodes
        return res

    def maxDepth(self, root: Optional[TreeNode]) -> int:
        """
        104.二叉树的最大深度
        2023.06.07 简单
        题解：最先想到的是层序遍历
        """
        if not root:
            return 0
        queue = [root]
        res = []
        while queue:
            tmpVals = []
            tmpNodes = []
            while queue:    # 耗尽一层的节点
                node = queue.pop(0)
                if node:
                    tmpVals.append(node.val)
                if node.left:
                    tmpNodes.append(node.left)
                if node.right:
                    tmpNodes.append(node.right)
            res.append(tmpVals)
            queue = tmpNodes
        return len(res)     # 返回的层数

    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        """
        105.从前序遍历与中序遍历序列构造二叉树
        2023.06.07 中等
        题解：递归 参数：指示同一子树的先序起终点索引、中序起终点索引
        """
        key2ind_inorder = {key: ind for ind, key in enumerate(inorder)}     # 哈希表，加速查找

        def fun(preorder_left, preorder_right, inorder_left, inorder_right):
            if preorder_left > preorder_right:      # 递归终止条件
                return None
            root_idx_inorder = key2ind_inorder[preorder[preorder_left]]   # fun()参数所指子树中，根节点所在中序遍历索引
            root = TreeNode(preorder[preorder_left])
            left_subtree_len = root_idx_inorder - inorder_left      # 计算fun()参数所指子树 的 左子树长度
            root.left = fun(preorder_left + 1, preorder_left + left_subtree_len,
                            inorder_left, root_idx_inorder - 1)
            root.right = fun(preorder_left + left_subtree_len + 1, preorder_right,
                             root_idx_inorder + 1, inorder_right)
            return root

        n = len(preorder)
        return fun(0, n - 1, 0, n - 1)

    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        114.二叉树展开为链表
        2023.06.08 中等
        题解：按题目要求展开就是先序遍历，所以总体思路就是不断地把“左子树”放到“右子树”的位置，而“右子树”放到“左子树”的右下，这样才能达到先序遍历的效果
            root的左子树放到右子树的位置，右子树放到刚才左子树的最右下
        """
        if not root:
            return
        while root:
            if not root.left:       # 当前节点没有左子树
                root = root.right
            else:                   # 当前节点有左子树
                rootLeft = root.left
                while rootLeft.right:   # 寻找左子树的最右下
                    rootLeft = rootLeft.right
                rootLeft.right = root.right     # 将右子树挂到左子树的最右
                root.right = root.left          # 将左子树放到右子树的位置
                root.left = None
                root = root.right       # 继续操作下一个节点，只能是当前节点的右孩子

        # 再写一遍
        # if not root:
        #     return
        # while root:
        #     if not root.left:   # 没有左孩子，则直接处理右孩子
        #         root = root.right
        #     else:   # 有左孩子，则寻找其最右下节点
        #         tmp = root.left     # tmp从左孩子寻找最右下节点
        #         while tmp.right:
        #             tmp = tmp.right
        #         tmp.right = root.right
        #         root.right = root.left
        #         root.left = None
        #         root = root.right

    def maxProfit(self, prices: List[int]) -> int:
        """
        121.买卖股票的最佳时机
        2023.06.08 简单
        题解：一次遍历
        """
        minVal = float('inf')
        maxProfit = 0
        for n in prices:
            if n < minVal:
                minVal = n
            if n - minVal > maxProfit:
                maxProfit = n - minVal
        return maxProfit

    def longestConsecutive(self, nums: List[int]) -> int:
        """
        128.最长连续序列
        2023.06.09 中等
        题解：暴力既不好实现、又不符合复杂度要求。
            答案，哈希大幅减少查找速度
        """
        numsSet = set(nums)     # 去重、哈希，加快查找速度
        longestLen = 0
        for n in numsSet:
            if n - 1 not in numsSet:
                curLen = 1      # 当前长度
                while n + 1 in numsSet:
                    n += 1
                    curLen += 1
                longestLen = max(longestLen, curLen)
        return longestLen

    def singleNumber(self, nums: List[int]) -> int:
        """
        136.只出现一次的数字
        2023.06.09 简单
        题解：异或
        """
        res = nums[0]
        for n in nums[1:]:
            res ^= n
        return res

        # 感觉不符合题意
        # import collections
        # key2counts = collections.Counter(nums)
        # for k, c in key2counts.items():
        #     if c == 1:
        #         return k

    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        """
        139.单词拆分
        2023.06.09 中等
        题解：动态规划
        """
        # 要注意 dp[i]的索引i 与 字符串s的索引[i:j]，两个i是有区别的
        # dp的i是指前i个字符，可以是0；s的i是字符串的起始索引，数值上两个i相同，意义上s的i要比dp的i往右一位，你品一下
        # n = len(s)
        # dp = [False for _ in range(n + 1)]      # 前 索引i 个字符组成的子串能否被wordDict表示
        # dp[0] = True
        # for i in range(n):
        #     for j in range(i + 1, n + 1):
        #         if dp[i] and s[i:j] in wordDict:
        #             dp[j] = True
        # return dp[-1]

        # 再写一遍
        n = len(s)
        dp = [False for _ in range(n + 1)]
        dp[0] = True
        for i in range(n):
            for j in range(i + 1, n + 1):
                if dp[i] and s[i:j] in wordDict:
                    dp[j] = True
        return dp[-1]

    def hasCycle(self, head: Optional[ListNode]) -> bool:
        """
        141.环形链表
        142.环形链表II
        2023.06.11 中等
        题解：答案，使用set。不必非得遍历完，遍历过程中只要发现了重复节点就返回True，否则返回False.
        """
        seen = set()
        while head:
            if head in seen:
                return True
            seen.add(head)
            head = head.next
        return False

    """ LRU缓存在文件最上面 """

    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        148.排序链表
        2023.06.11 中等
        题解：首先自己想的复杂度较高
            答案 归并排序 自顶向下，并非最优----时间复杂度O(nlogn) 空间复杂度O(logn)递归占用栈空间
            答案 归并排序 自底向上 代码太长了，但空间复杂度为O(1)，下次吧...
        """
        # 答案 归排 自顶向下
        # def mergeSort(head, tail):
        #     """
        #     为什么要两个参数嘞，因为要递归，递归的终止条件需要用头尾判断
        #     """
        #     if not head:    # 空
        #         return head
        #     if head.next == tail:   # 仅1个节点
        #         head.next = None
        #         return head
        #     # 使用快慢指针寻找链表的中间节点
        #     slow, fast = head, head
        #     while fast != tail:
        #         slow = slow.next
        #         fast = fast.next
        #         if fast != tail:
        #             fast = fast.next
        #     mid = slow      # 这里mid取slow或fast似乎都可以？？？不行，只能取slow，fast肯定到结尾了都，甚至可能是None
        #     head1 = mergeSort(head, mid)
        #     head2 = mergeSort(mid, tail)
        #     return merge(head1, head2)
        #
        # def merge(head1, head2):
        #     """ 将head1、head2视为两个独立的链表 """
        #     res = tmp = ListNode()
        #     while head1 and head2:
        #         if head1.val < head2.val:
        #             tmp.next = head1
        #             head1 = head1.next
        #         else:
        #             tmp.next = head2
        #             head2 = head2.next
        #         tmp = tmp.next
        #     if head1:
        #         tmp.next = head1
        #     if head2:
        #         tmp.next = head2
        #     return res.next
        #
        # return mergeSort(head, None)

        # 自己想的 复杂度较高
        # if not head:
        #     return None
        # mydict = {}
        # while head:
        #     mydict[head] = head.val
        #     head = head.next
        # mydict = sorted(mydict, key=lambda x: x.val)
        #
        # tmp = res = ListNode()
        # for node in mydict:
        #     tmp.next = ListNode(node.val)
        #     tmp = tmp.next
        # return res.next

        # 再写一遍，自顶向下 时间O(nlogn)、空间O(logn)递归
        def mergeSort(head, tail):
            """ head、tail参数，为什么比普通的归并排序多了tail参数，可能为了处理只有0或1个节点的特殊情况？
                                我想是的，因为普通的归并排序开始也判断了参数元素数量并特殊处理 """
            if not head:    # 无节点
                return head
            elif head.next == tail:     # 仅1个节点
                head.next = None
                return head
            # 使用快慢指针定位 中间节点
            slow = fast = head
            while fast != tail:     # 只要快指针没到结尾就继续遍历
                slow = slow.next
                fast = fast.next
                if fast != tail:
                    fast = fast.next
            mid = slow      # mid绝不可以等于fast，因为fast已经到链表尾了
            # 归并排序的经典递归写法
            head_1 = mergeSort(head, mid)
            head_2 = mergeSort(mid, tail)
            return merge(head_1, head_2)

        def merge(head_1, head_2):
            """
            合并两个链表：思路和普通的归并排序相同，将head_1、head_2视为两个独立的链表处理即可
            """
            resHead = tmp = ListNode()      # 头结点
            while head_1 and head_2:
                if head_1.val < head_2.val:
                    tmp.next = ListNode(head_1.val)
                    head_1 = head_1.next
                else:
                    tmp.next = ListNode(head_2.val)
                    head_2 = head_2.next
                tmp = tmp.next
            if head_1:
                tmp.next = head_1
            elif head_2:
                tmp.next = head_2
            return resHead.next

        return mergeSort(head, None)

    def maxProduct(self, nums: List[int]) -> int:
        """
        152.乘积最大子数组
        2023.06.11 中等
        题解：起码能一下想到-->动态规划
            答案，遍历 维护最大值、最小值(遇到负数，则最小值变最大值、最大值变最小值)
        """
        imin = imax = 1
        res = float('-inf')
        for n in nums:
            if n < 0:
                imin, imax = imax, imin
            imin = min(n, imin * n)     # 这是关键呀
            imax = max(n, imax * n)
            res = max(res, imax)
        return res

    """ 155.最小栈 写在最上面"""

    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        """
        160.相交链表
        2023.06.12 简单
        题解：那就按简单的思路来--效率极低
            答案 双指针
        """
        # 答案，双指针：二者相同的部分c个节点，不同的部分分别为a、b个节点，当二者遍历a + c + b次、b + c + a次时就能得出结果
        pA, pB = headA, headB
        while pA != pB:     # 自己画图看一下，相等的时候要么是相交节点要么是空
            pA = headB if pA is None else pA.next
            pB = headA if pB is None else pB.next
        return pA


        # 想法简单，遍历，效率极低，险过
        # lsA = []
        # while headA:
        #     lsA.append(headA)
        #     headA = headA.next
        # while headB:
        #     if headB in lsA:
        #         return headB
        #     headB = headB.next
        # return None

    def majorityElement(self, nums: List[int]) -> int:
        """
        169.多数元素
        2023.06.12 中等
        题解：堆排，时间复杂度：O(nlogn)，建堆O(n)  空间复杂度O(1)，哨兵
        """
        def HeapSort(nums):
            BuildHeap(nums)
            for i in range(len(nums) - 1, 0, -1):
                # print(nums[1], end=' ')
                nums[i], nums[1] = nums[1], nums[i]     # 由此可知，堆排序是可以直接返回排序后的结果的，不必一个一个的输出
                AdjustDown(nums, 1, i - 1)

        def BuildHeap(nums):
            i = len(nums) // 2
            while i > 0:
                AdjustDown(nums, i, len(nums) - 1)      # 建堆，有效的堆范围均为len(nums) - 1
                i -= 1

        def AdjustDown(nums, k, length):
            nums[0] = nums[k]
            i = 2 * k
            while i <= length:
                if i < length and nums[i] > nums[i + 1]:    # 我们从小到大输出
                    i = i + 1
                if nums[i] < nums[0]:       # 因为是待调整节点nums[k]本身不断往下调整，调整的目的是nums[k]而不是整个子树
                    nums[k] = nums[i]
                    k = i       # k的作用是记录最后nums[0]该放到哪里
                i *= 2      # nums[i]无论与其根节点换不换，索引i都得往下走
            nums[k] = nums[0]

        nums.insert(0, -1)      # 因为堆排序需要一个“哨兵”，因此添加一个哨兵
        HeapSort(nums)
        nums.pop(0)
        return nums[len(nums) // 2]

    def rob(self, nums: List[int]) -> int:
        """
        198.打家劫舍
        2023.06.13 中等
        题解：首先想到动态规划
        """
        dp = [-1] * (len(nums) + 1)
        dp[0] = 0
        dp[1] = nums[0]
        for i in range(2, len(nums) + 1):
            dp[i] = max(nums[i - 1] + dp[i - 2], dp[i - 1])
        return dp[-1]

    def numIslands(self, grid: List[List[str]]) -> int:
        """
        200.岛屿数量
        2023.06.13 中等
        题解：答案 深度遍历，见1就认为是岛屿并且深度遍历并将其置0
        """
        def dfs(x, y):
            grid[x][y] = "0"      # 见1置零
            for x, y in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
                if 0 <= x < nr and 0 <= y < nc and grid[x][y] == "1":
                    dfs(x, y)       # 递归，深度遍历

        nr = len(grid)      # 题目已说明行列>=1，所以不用判断==0
        nc = len(grid[0])
        islandNum = 0
        for x in range(nr):
            for y in range(nc):
                if grid[x][y] == "1":
                    islandNum += 1
                    dfs(x, y)
        return islandNum

    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        206.反转链表
        2023.06.13 中等
        题解：首先拒绝遍历
            答案 迭代
        """
        prev, curr = None, head
        while curr:
            next = curr.next
            curr.next = prev
            prev = curr
            curr = next
        return prev

if __name__ == '__main__':
    sl = Solution()

    nums = [2, 7, 9, 3, 1]
    print(sl.rob(nums))

    # from random import randint
    # nums = [randint(0, 20) for _ in range(10)]
    # print(nums)
    # sl.majorityElement(nums)
    # print(nums)

    # nums = [2,3,-2,4]
    # print(sl.maxProduct(nums))

    # prices = [7, 6, 4, 3, 1]
    # print(sl.maxProfit(prices))

    # n = 1
    # print(sl.numTrees(n))

    # board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
    # word = "ABCCED"
    # print(sl.exist(board, word))

    # nums = [0]
    # print(sl.subsets(nums))

    # nums = [random.randint(0, 3) for _ in range(10)]
    # print(nums)
    # print(sl.sortColors(nums))
    # print(nums)

    # print(sl.climbStairs(3))

    # grid = [[1,2,3],[4,5,6]]
    # print(sl.minPathSum(grid))

    # print(sl.uniquePaths(3, 3))

    # intervals = [[1,4],[4,5]]
    # print(sl.merge(intervals))

    # nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    # print(sl.maxSubArray(nums))

    # strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
    # print(sl.groupAnagrams(strs))
    # matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    # sl.rotate(matrix)
    # print(matrix)

    # nums = [1]
    # target = 0
    # print(sl.search(nums, target))

    # nums = [1, 3, 2]
    # sl.nextPermutation(nums)
    # print(nums)

    # print(sl.generateParenthesis(3))
    # s = "(){}}{"
    # print(sl.isValid(s))

    # digits = "23"
    # print(sl.letterCombinations(digits))

    # l1 = [2, 4, 3]
    # l2 = [5, 6, 4]
    # print(sl.addTwoNumbers(l1, l2))