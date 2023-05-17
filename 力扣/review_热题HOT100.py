import collections
from typing import List, Optional

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        """
        55.跳跃游戏
        :param nums:
        :return:
        """
        max_i = 0
        for i, jump in enumerate(nums):
            if i <= max_i < i + jump:
                max_i = i + jump
        return True if max_i >= len(nums) - 1 else False

    def maxSubArray(self, nums: List[int]) -> int:
        """
        53.最大子数组和
        :param nums:
        :return:
        """
        sum_ind = [0 for i in range(len(nums))]
        sum_ind[0] = nums[0]
        for i in range(1, len(nums)):
            if sum_ind[i - 1] > 0:
                sum_ind[i] = sum_ind[i - 1] + nums[i]
            else:
                sum_ind[i] = nums[i]
        return max(sum_ind)

    def canJump01(self, nums: List[int]) -> bool:
        max_range = 0
        for i, jump in enumerate(nums):
            if i <= max_range <= i + jump:
                max_range = i + jump
        return max_range >= len(nums) - 1

    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        """
        49.字母异位词分组
        :param strs:
        :return:
        """
        dict_Ana = collections.defaultdict(list)
        for word in strs:
            dict_Ana[''.join(sorted(word))].append(word)
        return list(dict_Ana.values())

    def rotate(self, matrix: List[List[int]]) -> None:
        """
        48.旋转图像
        :param matrix:
        :return:
        """
        matrix_new = [[0] * len(matrix) for i in range(len(matrix))]
        for row in range(len(matrix)):
            for col in range(len(matrix)):
                matrix_new[col][-row - 1] = matrix[row][col]
        matrix[:] = matrix_new

    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        """
        47.全排列II
        2022.11.25
        :param nums:
        :return:
        """
        def dfs(nums, depth, size, path, res, used):
            if depth == size:
                if path not in res:
                    res.append(path[:])
                return

            for i in range(size):
                if not used[i]:
                    path.append[nums[i]]
                    used[i] = True

                    dfs(nums, depth + 1, size, path, res, used)
                    path.pop()
                    used[i] = False

        size = len(nums)
        path, res = [], []
        used = [False for _ in range(size)]
        dfs(nums, 0, size, path, res, used)
        return res

    def permute(self, nums: List[int]) -> List[List[int]]:
        """
        46.全排列
        2022.12.03
        :param nums:
        :return:
        """
        def dfs(nums, size, depth, used, path, res):
            if depth == size:
                res.append(path[:])
                return
            for i in range(size):
                if not used[i]:
                    path.append(nums[i])
                    used[i] = True
                    dfs(nums, size, depth + 1, used, path, res)
                    path.pop()
                    used[i] = False

        size = len(nums)
        if not size:
            return []
        path, res = [], []
        used = [False for _ in range(size)]
        dfs(nums, size, 0, used, path, res)
        return res

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        """
        39.组合总和
        2022.12.04
        :param candidates:
        :param target:
        :return:
        """
        def dfs(nums, ind, size, path, res, target):
            if target < 0:
                return
            if target == 0:
                res.append(path)
                return
            for i in range(ind, size):
                dfs(nums, i, size, path + [nums[i]], res, target - nums[i])

        begin, size = 0, len(candidates)
        path, res = [], []
        dfs(candidates, begin, size, path, res, target)
        return res

    def searchRange(self, nums: List[int], target: int) -> List[int]:
        """
        34.在排序数组中查找元素的第一个和最后一个位置
        2022.12.04
        :param nums:
        :param target:
        :return:
        """
        pos = -1
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = (l + r) // 2
            if nums[mid] == target:
                pos = mid
                break
            elif nums[mid] < target:
                l = mid + 1
            else:
                r = mid - 1
        if pos == -1:
            return [-1, -1]
        left = pos - 1
        while left >= 0 and nums[left] == target:
            left -= 1
        left += 1
        right = pos + 1
        while right <= len(nums) - 1 and nums[right] == target:
            right += 1
        right -= 1
        return [left, right]

    def search(self, nums: List[int], target: int) -> int:
        """
        33.搜索旋转排序数组
        2022.12.05
        :param nums:
        :param target:
        :return:
        """
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = (l + r) // 2
            if nums[mid] == target:
                return mid
            elif nums[l] <= nums[mid]:      # 这个等号可是费劲了——万一左边是单个元素，没有考虑，右边也不是升序
                if nums[l] <= target < nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1
            else:
                if nums[mid] < target <= nums[r]:
                    l = mid + 1
                else:
                    r = mid - 1
        return -1

    def nextPermutation(self, nums: List[int]) -> None:
        """
        31.下一个排列
        2022.12.06
        :param nums:
        :return:
        """
        if len(nums) <= 2:
            nums.sort()

        flag = False
        for i in range(len(nums) - 1, 0, -1):
            if nums[i - 1] < nums[i]:
                flag = True
                for k in range(len(nums) - 1, i - 1, -1):
                    if nums[k] > nums[i - 1]:
                        nums[k], nums[i - 1] = nums[i - 1], nums[k]
                        break
                nums[i:] = sorted(nums[i:])[:]      # 注意深拷贝
                break
        if not flag:
            nums.sort()

    def removeElement(self, nums: List[int], val: int) -> int:
        """
        27.移除元素
        2022.12.07
        :param nums:
        :param val:
        :return:
        """
        last = 0
        for i in range(len(nums)):
            if nums[i] != val:
                nums[last] = nums[i]
                last += 1
        return len(nums[:last])

    def removeDuplicates(self, nums: List[int]) -> int:
        """
        26.删除有序数组中的重复项
        2022.12.09
        :param nums:
        :return:
        """
        if len(nums) == 1:
            return 1

        i, j = 0, 1
        while j < len(nums):
            if nums[i] != nums[j]:
                nums[i + 1] = nums[j]
                i += 1
            j += 1
        return len(nums[:i + 1])


    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        24.两两交换链表中的节点
        2022.12.15
        :param head:
        :return:
        """
        if not head or not head.next:
            return head

        cur = head
        stack = [cur, cur.next]
        cur = cur.next.next
        head = stack.pop()
        head.next = stack.pop()
        pre = head.next

        while cur and cur.next:
            stack.append(cur)
            stack.append(cur.next)
            cur = cur.next.next

            pre.next = stack.pop()
            pre.next.next = stack.pop()
            pre = pre.next.next

        if cur and not cur.next:
            pre.next = cur
            pre = pre.next
        pre.next = None
        return head

    def generateParenthesis(self, n: int) -> List[str]:
        """
        22.括号生成
        中等 2022.09.26
        题解："剩下的括号要么在新增括号里面，要么右边"--看题目示例一就能明白了。
        题目要求1<=n，所以不用单独考虑返回n=0时的情况
        :param n:
        :return:
        """
        # [None]表示列表里面有元素，[]就没元素了；之所以[None]是因为p q两个循环的时候不退出，for k in []会直接退出
        # res_n = [[None], ['()']]
        # for i in range(2, n + 1):
        #     tmp_i = []
        #     for j in range(i):
        #         list_p = res_n[j]
        #         list_q = res_n[i - 1 - j]
        #         for k1 in list_p:
        #             for k2 in list_q:
        #                 k1 = "" if not k1 else k1
        #                 k2 = "" if not k2 else k2
        #                 element = '(' + k1 + ')' + k2
        #                 tmp_i.append(element)
        #     res_n.append(tmp_i)
        # return res_n[n]

        res_n = [[None], ['()']]    # n=0、1时结果
        for i in range(2, n + 1):   # 依次计算n=2、...、n的结果，并放入res_n
            tmp_i = []      # 计算n=i时的结果
            for j in range(i):  # 已知n=i-1的结果，计算n=i的结果
                list_q = res_n[j]
                list_p = res_n[i - 1 - j]   # 二者索引相加要==i - 1
                for k1 in list_q:
                    for k2 in list_p:
                        k1 = '' if not k1 else k1
                        k2 = '' if not k2 else k2
                        element = '(' + k1 + ')' + k2
                        tmp_i.append(element)
            res_n.append(tmp_i)
        return res_n[n]

    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        """
        2022.12.26
        21.合并两个有序链表
        :param list1:
        :param list2:
        :return:
        """
        res = head = ListNode()
        while list1 and list2:
            if list1.val < list2.val:
                head.next = list1
                list1 = list1.next
            else:
                head.next = list2
                list2 = list2.next
            head = head.next
        while list1:
            head.next = list1
            head = head.next
            list1 = list1.next
        while list2:
            head.next = list2
            head = head.next
            list2 = list2.next
        return res.next

    def isValid(self, s: str) -> bool:
        """
        20.有效的括号
        2022.12.28
        :param s:
        :return:
        """
        # stack = []
        # while s:
        #     if stack and (s[0] == ')' and stack[-1] == '(' or
        #                   s[0] == ']' and stack[-1] == '[' or
        #                   s[0] == '}' and stack[-1] == '{'):
        #         stack.pop(-1)
        #         s = s[1:]
        #     else:
        #         stack.append(s[0])
        #         s = s[1:]
        # return True if len(stack) == 0 and len(s) == 0 else False

        bracket_dict = {'(': ')', '{': '}', '[': ']'}
        stack = []      # key才入栈
        for c in s:
            if c in bracket_dict.keys():
                stack.append(c)
            elif len(stack) and c == bracket_dict[stack[-1]]:
                stack.pop(-1)
            else:
                return False
        return True if not len(stack) else False

    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        """
        19.删除链表的倒数第N个节点
        2022.12.29
        :param head:
        :param n:
        :return:
        """
        total = 0
        cur = head
        while cur:
            total += 1
            cur = cur.next

        pre = total - n     # 正数的前1个节点
        if pre < 1:
            return head.next
        cur = head
        while cur and pre > 1:
            pre -= 1
            cur = cur.next
        cur.next = cur.next.next
        return head

    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        """
        18.四数之和
        2023.01.02
        :param nums:
        :param target:
        :return:
        """
        nums.sort()
        res = set()
        for i in range(len(nums) - 3):
            for j in range(i + 1, len(nums) - 2):
                l, r = j + 1, len(nums) - 1
                while l < r:
                    if nums[i] + nums[j] + nums[l] + nums[r] == target:
                        res.add((nums[i], nums[j], nums[l], nums[r]))
                        l += 1
                    elif nums[i] + nums[j] + nums[l] + nums[r] < target:
                        l += 1
                    else:
                        r -= 1
        return [list(e) for e in res]

    def letterCombinations(self, digits: str) -> List[str]:
        """
        17.电话号码的字母组合
        :param digits:
        :return:
        """
        digit_dict = {2: list('abc'), 3: list('def'), 4: list('ghi'), 5: list('jkl'),
                      6: list('mno'), 7: list('pqrs'), 8: list('tuv'), 9: list('wxyz')}
        if len(digits) == 0:
            return []
        elif len(digits) == 1:
            return digit_dict[int(digits)]

        res = digit_dict[int(digits[0])]    # list, 元素是字符串
        for d in digits[1:]:
            res = [sub_r + c for sub_r in res for c in digit_dict[int(d)]]
        return res

    def threeSumClosest(self, nums: List[int], target: int) -> int:
        """
        16.最接近的三数之和
        2022.01.03
        :param nums:
        :param target:
        :return:
        """
        nums.sort()
        cur_sum = sum(nums[:3])     # 以 前3个数的和 初始化。因为，假如初始化的值就是最接近的而nums中并没有这样的三个数，就错了
        for i in range(len(nums) - 2):
            l, r = i + 1, len(nums) - 1
            while l < r:
                tmp_sum = nums[i] + nums[l] + nums[r]
                cur_sum = tmp_sum if abs(tmp_sum - target) < abs(cur_sum - target) else cur_sum     # 及时更新cur_sum
                if tmp_sum == target:
                    return tmp_sum
                elif tmp_sum < target:
                    l += 1
                elif tmp_sum > target:
                    r -= 1
        return cur_sum

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        """
        15.三数之和
        2022.01.03
        :param nums:
        :return:
        """
        res = set()
        nums.sort()
        for i in range(len(nums) - 2):
            l, r = i + 1, len(nums) - 1
            while l < r:
                tmp_sum = nums[i] + nums[l] + nums[r]
                if tmp_sum == 0:
                    res.add((nums[i], nums[l], nums[r]))
                    l += 1
                elif tmp_sum < 0:
                    l += 1
                elif tmp_sum > 0:
                    r -= 1
        return [list(tup) for tup in res]

    def longestCommonPrefix(self, strs: List[str]) -> str:
        """
        14.最长公共前缀
        2023.01.04
        :param strs:
        :return:
        """
        if len(strs) == 1:
            return strs[0]

        res = ""
        for cs in zip(*strs):
            first = cs[0]
            for c in cs[1:]:
                if first != c:
                    return res
            res += first
        return res

    def romanToInt(self, s: str) -> int:
        """
        13.罗马数字转整数
        2023.01.05
        :param s:
        :return:
        """
        Rome_dict = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        if len(s) == 1:
            return Rome_dict[s]
        digit = 0
        i, j = 0, 1     # 从左往右处理
        while j < len(s):
            if Rome_dict[s[i]] >= Rome_dict[s[j]]:  # 大->小 顺序，加上i指向的值
                digit += Rome_dict[s[i]]
            else:   # 小->大 顺序，减去i指向的值
                digit -= Rome_dict[s[i]]
            i += 1
            j += 1
        digit += Rome_dict[s[j - 1]]
        return digit

    def intToRoman(self, num: int) -> str:
        """
        12.整数转罗马数字
        2023.01.05
        :param num:
        :return:
        """
        Rome_dict = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000,
                     'IV': 4, 'IX': 9, 'XL': 40, 'XC': 90, 'CD': 400, 'CM': 900}
        Rome_dict = dict(sorted(Rome_dict.items(), key=lambda x: x[1], reverse=True))
        rome = ''
        while num:
            for k, v in Rome_dict.items():
                while num / v >= 1:
                    num -= Rome_dict[k]
                    rome += k
        return rome

    def maxArea(self, height: List[int]) -> int:
        """
        11.盛最多水的容器
        2023.01.09
        :param height:
        :return:
        """
        i, j = 0, len(height) - 1
        max_area = min(height[i], height[j]) * (j - i)
        while i < j:
            if height[i] < height[j]:
                i += 1
            else:
                j -= 1
            area = min(height[i], height[j]) * (j - i)
            max_area = area if area > max_area else max_area
        return max_area

    def longestPalindrome(self, s: str) -> str:
        """
        5.最长回文子串
        :param s:
        :return:
        """
        res = [0, 0]
        for i in range(len(s)):
            left = right = i
            while left >= 0 and s[left] == s[i]:
                left -= 1
            left += 1
            while right <= len(s) - 1 and s[right] == s[i]:
                right += 1
            right -= 1

            while left >= 0 and right <= len(s) - 1 and s[left] == s[right]:
                left -= 1
                right += 1
            if res[1] - res[0] < right - 1 - left - 1:
                res = (left + 1, right - 1)
        return s[res[0]:res[1] + 1]

    def isPalindrome(self, x: int) -> bool:
        """
        9.回文数
        2023.01.10
        :param x:
        :return:
        """
        # x = str(x)
        # if x == "".join(reversed(x)):
        #     return True
        # return False

        if x < 0:
            return False
        x_copy = x
        rev = 0
        while x_copy > 0:
            rev = rev * 10 + x_copy % 10
            x_copy //= 10
        return True if rev == x else False

    def myAtoi(self, s: str) -> int:
        """
        8.字符串转整数
        :param s:
        :return:
        """
        # 参考的答案 这无论如何都会遍历一遍，但是代码精简不少
        flag = False
        res = 0
        for i in range(len(s)):
            if s[i] == '-':
                flag = not flag
            if '0' <= s[i] <= '9':
                res = res * 10 + ord(s[i]) - ord('0')
        return res if not flag else res * -1

        # 太长了，且多个地方需要考虑
        # s = s.lstrip()      # 似乎不用去掉右边
        # if len(s) == 0:
        #     return 0
        #
        # neg = 0
        # if s[0] == '-':
        #     neg += 1
        #     s = s[1:]
        # elif s[0] == '+':
        #     s = s[1:]
        # if len(s) == 0:
        #     return 0
        # legal_ls = list("0123456789")
        # result = 0
        # while s[0] in legal_ls:
        #     result = result * 10 + int(s[0])
        #     s = s[1:]
        #     if len(s) == 0:
        #         break
        # if neg:
        #     result = -2**31 if result * -1 < -2**31 else result * -1
        # else:
        #     result = 2**31 - 1 if result > 2**31 - 1 else result
        #
        # return result

    def reverse(self, x: int) -> int:
        if x == 0:
            return 0
        neg = 0
        if x < 0:
            neg = 1
            x *= -1
        x = list(str(x))
        x.reverse()
        res = 0
        while x:
            res = res * 10 + ord(x[0]) - ord('0')
            if len(x) == 1:
                break
            x = x[1:]
        res = res if not neg else res * -1
        res = 0 if res < -2**31 or res > 2**31 - 1 else res
        return res

    def convert(self, s: str, numRows: int) -> str:
        """
        6.Z字形变换
        :param s:
        :param numRows:
        :return:
        """
        if numRows == 1:
            return s

        str_rows = ['' for _ in range(numRows)]
        cur = 0
        flag = False    # 正向/反向轮询字符串
        while s:
            str_rows[cur] += s[0]
            if len(s) == 1:
                break
            s = s[1:]

            if cur == numRows - 1:
                flag = True
            elif cur == 0:
                flag = False

            if not flag:
                cur += 1
            else:
                cur -= 1

        res = ""
        for e in str_rows:
            res += e
        return res

    def longestPalindrome1(self, s: str) -> str:
        """
        5.最长回文子串
        :param s:
        :return:
        """
        res = ''
        for i in range(len(s)):
            if i - 1 >= 0 and s[i - 1] == s[i]:
                left, right = i - 1 - 1, i + 1
                while left >= 0 and right <= len(s) - 1 and s[left] == s[right]:
                    left -= 1
                    right += 1
                res = s[left + 1:right] if len(s[left + 1:right]) > len(res) else res
            elif i + 1 <= len(s) - 1 and s[i + 1] == s[i]:
                left, right = i - 1, i + 1 + 1
                while left >= 0 and right <= len(s) - 1 and s[left] == s[right]:
                    left -= 1
                    right += 1
                res = s[left + 1:right] if len(s[left + 1:right]) > len(res) else res

            left, right = i - 1, i + 1
            while left >= 0 and right <= len(s) - 1 and s[left] == s[right]:
                left -= 1
                right += 1
            res = s[left + 1:right] if len(s[left + 1:right]) > len(res) else res
        return res

    def lengthOfLongestSubstring(self, s: str) -> int:
        """
        3.无重复字符的最长子串
        :param s:
        :return:
        """
        if len(s) <= 1:
            return len(s)

        i, j = 0, 1
        res = 0
        while j <= len(s) - 1:
            if s[j] in s[i:j]:
                i += s[i:j].index(s[j]) + 1
            res = len(s[i:j + 1]) if len(s[i:j + 1]) > res else res
            j += 1
        return res

    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        """
        2.两数相加
        :param l1:
        :param l2:
        :return:
        """
        num1 = num2 = ""
        while l1:
            num1 = str(l1.val) + num1
            l1 = l1.next
        while l2:
            num2 = str(l2.val) + num2
            l2 = l2.next
        res = int(num1) + int(num2)
        head = cur = ListNode(res % 10)
        res //= 10
        while res:
            cur.next = ListNode(res % 10)
            res //= 10
            cur = cur.next
        return head

    def twoSum(self, nums: List[int], target: int) -> List[int]:
        """
        1.两数之和
        题解：https://leetcode.cn/problems/two-sum/solution/xiao-bai-pythonji-chong-jie-fa-by-lao-la-rou-yue-j/
            题目要找num1 + num2 = target，遍历看看nums数组里有没有target - num1：
                1.有，再看找到的是不是num1本身，即target - num1 是不是等于 num1
                2.没有，那就continue呗
            优化：你需要把两个数颠倒过来看, 也就是说他只有找到你说的 num2 时 才会在前面找到你说的num,也只有这个时候才输出结果,
                也就是说两个数都被遍历到时才会找到答案, a + b 和 b + a 不是相等的吗
        :param nums:
        :param target:
        :return:
        """
        # 方法二：优化方法一
        # j = -1
        for i in range(1, len(nums)):
            prior = nums[:i]
            if target - nums[i] in prior:
                return [prior.index(target - nums[i]), i]
        return []


        # 方法一
        # j = -1
        # for i in range(len(nums)):
        #     if nums.count(target - nums[i]) == 1 and target - nums[i] == nums[i]:   # 有戏，再看找到的是不是本身
        #         continue
        #     elif nums.count(target - nums[i]):
        #         j = nums.index(target - nums[i], i + 1)
        #         break
        # return [i, j] if j != -1 else []

if __name__ == '__main__':
    sl = Solution()
    nums = [3,3]
    target = 6
    print(sl.twoSum(nums, target))

    # print(sl.lengthOfLongestSubstring("pwwkew"))

    # print(sl.longestPalindrome1('cbbd'))

    # print(sl.convert("PAYPALISHIRING", 1))

    # x = 0
    # print(sl.reverse(x))

    # s = "  4193 with words"
    # print(sl.myAtoi(s))

    # s = 'cbbd'
    # print(sl.longestPalindrome(s))

    # print(sl.isPalindrome(121))

    # print(sl.maxArea([1,1]))

    # print(sl.intToRoman(1994))

    # s = 'MCMXCIV'
    # print(sl.romanToInt(s))

    # strs = ["a"]
    # print(sl.longestCommonPrefix(strs))

    # nums = [-1, 0, 1, 2, -1, -4]
    # print(sl.threeSum(nums))

    # nums = [0, 0, 0]
    # target = 1
    # print(sl.threeSumClosest(nums, target))
    # digits = "23"
    # print(sl.letterCombinations(digits))

    # nums = [2,2,2,2,2]
    # target = 8
    # print(sl.fourSum(nums, target))

    # ListNode
    # nums = [1, 2]
    # cur = head = ListNode(nums[0])
    # for n in nums[1:]:
    #     cur.next = ListNode(n)
    #     cur = cur.next
    #
    # head = sl.removeNthFromEnd(head, 2)
    # while head:
    #     print(head.val)
    #     head = head.next


    # s = "()[]{}"
    # print(sl.isValid(s))

    # print(sl.generateParenthesis(1))

    # nums = []
    # head = ListNode(nums.pop(0))
    # cur = head
    # while nums:
    #     cur.next = ListNode(nums.pop(0))
    #     cur = cur.next

    # while head:
    #     print(head.val)
    #     head = head.next

    # head = sl.swapPairs(ListNode(-1))
    # while head:
    #     print(head.val)
    #     head = head.next


    # nums = [1,1,2]
    # len = sl.removeDuplicates(nums)
    # print(len)

    # nums = [1, 3, 2]
    # sl.nextPermutation(nums)
    # print(nums)

    # nums = [3, 1]
    # target = 1
    # print(sl.search(nums, target))

    # nums = [5,7,7,8,8,10]
    # target = 6
    # print(sl.searchRange(nums, target))

    # candidates = [2, 3, 5]
    # target = 8
    # print(sl.combinationSum(candidates, target))

    # matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
    # print(sl.rotate(matrix))

    # strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
    # print(sl.groupAnagrams(strs))

    # nums = [5,4,-1,7,8]
    # print(sl.maxSubArray(nums))

    # nums = [3,2,1,0,4]
    # print(sl.canJump01(nums))