import collections
from typing import List, Optional


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        """
        1.两数之和
        简单
        2022.09.18
        :param nums:整数数组
        :param target:整数目标值
        :return:和为target的两个整数下标 不能重复
        """
        for i in range(len(nums) - 1):
            for j in range(i + 1, len(nums)):
                if nums[i] + nums[j] == target:
                    print(f'下标为{[i, j]}')
                    return [i, j]

    # Definition for singly-linked list.
    class ListNode:
        def __init__(self, val=0, next=None):
            self.val = val
            self.next = next

    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        """
        2.两数相加
        中等
        2022.09.18
        :param l1: 链表，每个节点存储一个数的一位。如，234存储为4->3->2
        :param l2:
        :return:
        """
        num1, num2 = '', ''
        while l1:
            num1 = str(l1.val) + num1
            l1 = l1.next
        while l2:
            num2 = str(l2.val) + num2
            l2 = l2.next
        print(f'num1:{int(num1)}, num2:{num2}')  # str-->int
        sum = int(num1) + int(num2)
        head = self.ListNode(sum % 10)
        cur = head
        sum //= 10
        while sum:
            cur.next = self.ListNode(sum % 10)
            cur = cur.next
            sum //= 10
        return head

    def lengthOfLongestSubstring(self, s: str) -> int:
        """
        3.无重复字符的最长子串
        中等
        2022.09.18
        题解：滑动窗口的思想，排除考虑长度<2的情况，j慢慢向右探测，如果j指向的元素e在s[i:j]中也有，则i跳到s[i:j]中e元素后一位
        :param s:
        :return:
        """
        if len(s) < 2:
            return len(s)
        i, j = 0, 1
        lengths = []
        while j < len(s):
            if s[j] in s[i:j]:
                ind = s[i:j].index(s[j]) + i
                i = ind + 1
            # print(f'长度:{j - i + 1}')
            lengths.append(j - i + 1)
            j += 1
        # print(f'最长:{max(lengths)}')
        return max(lengths)

    def longestPalindrome(self, s: str) -> str:
        """
        5.最长回文子串
        中等
        题解：https://leetcode.cn/problems/longest-palindromic-substring/solution/zhong-xin-kuo-san-fa-he-dong-tai-gui-hua-by-reedfa/
            中心扩散法。对每一个元素a，分别向两边寻找与a相同的元素 或 s[i]==s[j]，记录回文子串长度和起始位置
        2022.09.18
        :param s:
        :return:
        """
        maxlen = 1  # 记录最长回文子串的长度
        start = 0  # 记录最长回文子串起始位置
        for i in range(len(s)):
            left = i - 1
            right = i + 1
            len_sub = 1  # 回文子串的长度
            while left >= 0 and s[left] == s[i]:
                len_sub += 1
                left -= 1
            while right < len(s) and s[right] == s[i]:
                len_sub += 1
                right += 1
            while left >= 0 and right < len(s) and s[left] == s[right]:
                len_sub += 2
                left -= 1
                right += 1
            if len_sub > maxlen:
                maxlen = len_sub
                start = left + 1
        return s[start:start + maxlen]

    def convert(self, s: str, numRows: int) -> str:
        """
        6.Z字形变换
        中等
        2022.09.19
        题解：建立一个二维数组盛放结果。效率较低...
            如果目标行数<=1时直接返回s即可。
            目标行数>1时需仔细考虑列数，因为不是所有的列都能填满，有的列元素数量是num_rows-2。
            从每一列上来回填充，并且碰到开始、结尾的元素就往反方向走。
        leetcode题解：使用numRows个字符串，依次存储各元素
        :param s:
        :param numRows:
        :return:
        """
        # leetcode思路:效率稍有提升。
        #   将结果保存在一个列表中--元素为numRows个字符串，将s中元素来回依次放入numRows个字符串中。
        #   记得特殊处理numRows<2
        if numRows < 2:
            return s
        x = iter(s)
        res = ['' for _ in range(numRows)]
        cur_row = 0
        while True:
            try:
                while cur_row <= numRows - 1:
                    res[cur_row] += next(x)
                    cur_row += 1
                cur_row = numRows - 2

                while cur_row >= 0:
                    res[cur_row] += next(x)
                    cur_row -= 1
                cur_row = 1
            except StopIteration:
                break
        save = ''
        for r in res:
            save += r
        return save

        # import math
        # import numpy as np
        #
        # if numRows <= 1:
        #     return s
        # numCols = 2 * math.ceil(len(s) / (2 * numRows - 2))   # 向上取整
        # matrix = np.array(np.zeros((numRows, numCols), dtype=int), dtype=str)
        # cur_row, cur_col = 0, 0
        # elements = iter(s)
        # while True:
        #     try:
        #         while cur_row < numRows:
        #             matrix[cur_row, cur_col] = next(elements)
        #             cur_row += 1
        #         cur_row = numRows - 2
        #         cur_col += 1
        #
        #         while cur_row >= 0:
        #             matrix[cur_row, cur_col] = next(elements)
        #             cur_row -= 1
        #         cur_row = 1
        #         cur_col += 1
        #     except StopIteration:
        #         break
        # # print(matrix)
        #
        # res = ''
        # for ele_row in matrix:
        #     for e in ele_row:
        #         res = res + e if e != '0' else res
        # return res

    def reverse(self, x: int) -> int:
        """
        7.整数反转
        中等 2022.09.20
        :param x:
        :return:
        """
        if x == 0:
            return 0

        x_str = str(x)
        flag = False
        if x_str[0] == '-':
            x_str = x_str[1:]
            flag = True
        x_str = x_str[::-1]
        while True:
            if x_str[0] == '0':
                x_str = x_str[1:]
            else:
                break
        res = int(x_str) if not flag else int(x_str) * -1
        return res if -(2 ** 31) <= res <= 2 ** 31 - 1 else 0

    def myAtoi(self, s: str) -> int:
        """
        8.字符串转换整数 (atoi)
        中等 2022.09.20
        问题：测试用例的"words and 987"预期输出是0？
        :param s:
        :return:
        """
        # 这代码似乎不符题意
        res = 0
        flag = False
        for i in range(len(s)):
            if s[i] == '-':
                flag = True
            if '0' <= s[i] <= '9':
                res = res * 10 + (ord(s[i]) - ord('0'))
        return res if not flag else res * -1

        # 测试用例的"words and 987"预期输出是0？——20230111对呀，前面没数字都不会开始遍历
        # import numpy as np
        #
        # allmywanted = list("0123456789")
        # s_bool = [True if ele in allmywanted else False for ele in s]
        # s_ind = np.array(s_bool)
        # s_np = np.array(list(s), dtype=str)
        # res = ''
        # for n in s_np[s_ind]:
        #     res += n
        # # print(s.index(res[0]))
        # start = s.index(res[0])
        # flag = False
        # if start > 0:
        #    if s[start - 1] == '-':
        #        flag = True
        #
        # return int(res) if not flag else int(res) * -1

    def isPalindrome(self, x: int) -> bool:
        """
        9.回文数
        简单 2022.09.22
        :param x:
        :return:
        """
        # 方法一：转换为字符串
        # s = str(x)
        # s_rev = s[::-1]
        # # print(f's:{s}\ts_rev:{s_rev}')
        # return True if s == s_rev else False

        # 方法二：
        if x < 0:
            return False
        num = x
        cur = 0
        while num > 0:
            cur = cur * 10 + num % 10
            num //= 10  # 这里用/的话可能会有小数，导致cur是个inf
        # print(f'x={x}\tcur={cur}')
        return x == cur

    def maxArea(self, height: List[int]) -> int:
        """
        11.盛最多水的容器
        中等 2022.09.22
        题解：leecode.
        无论是移动短板或者长板，我们都只关注移动后的新短板会不会变长，而每次移动的木板都只有三种情况，比原短板短，比原短板长，与原短板相等；
        如向内移动长板，对于新的木板：1.比原短板短，则新短板更短。2.与原短板相等或者比原短板长，则新短板不变。所以，向内移动长板，一定不能使新短板变长。
        :param height: 长度为n
        :return:
        """
        max_area = 0
        i, j = 0, len(height) - 1
        while i < j:
            area = (j - i) * min(height[i], height[j])
            max_area = area if area > max_area else max_area
            if height[i] < height[j]:
                i += 1
            else:
                j -= 1
        return max_area

    def intToRoman(self, num: int) -> str:
        """
        12.整数转罗马数字
        中等 2022.09.22
        题解：将所有情况从小到大写一个字典，对num的每一位计算，比如有几个‘M’呀，每一位有几个'CM'呀等等。有了思路也很简单
        :param num:
        :return:
        """
        hashmap = {'M': 1000, 'CM': 900, 'D': 500, 'CD': 400, 'C': 100, 'XC': 90, 'L': 50, 'XL': 40, 'X': 10, 'IX': 9,
                   'V': 5, 'IV': 4, 'I': 1}
        num_roman = ''
        while num > 0:
            for key, val in hashmap.items():
                while num >= val:
                    num_roman += key
                    num -= val
        return num_roman

    def romanToInt(self, s: str) -> int:
        """
        13.罗马数字转整数
        简单 2022.09.22
        题解：还是用上题的hashmap，对每个key看s中有几个就加几次val，但要先检查两位的key，先检查单位的key可能会覆盖
            我觉得不如我后来想出来的简单
        :param s:
        :return:
        """
        # hashmap = {'CM': 900, 'CD': 400, 'XC': 90, 'XL': 40, 'IX': 9, 'IV': 4, 'M': 1000, 'D': 500, 'C': 100, 'L': 50, 'X': 10,
        #            'V': 5, 'I': 1}
        # num = 0
        # for k, v in hashmap.items():
        #     while k in s:
        #         num += v
        #         start = s.index(k)
        #         s = s.replace(s[start:start+len(k)], '', 1)
        # return num

        # leecode解法，从右往左，小值_大值则减小值加大值，大值_小值都加；也可理解为遇到小值就减，否则一直加
        hashmap = {'M': 1000, 'CM': 900, 'D': 500, 'CD': 400, 'C': 100, 'XC': 90, 'L': 50, 'XL': 40, 'X': 10, 'IX': 9,
                   'V': 5, 'IV': 4, 'I': 1}
        s = s[::-1]
        num = hashmap[s[0]]
        for i in range(1, len(s)):
            if hashmap[s[i - 1]] <= hashmap[s[i]]:
                num += hashmap[s[i]]
            else:
                num -= hashmap[s[i]]
        return num

    def longestCommonPrefix(self, strs: List[str]) -> str:
        """
        14.最长公共前缀
        简单 2022.09.23
        题解：可能脑子瓦特，感觉并不简单，参考了一点题解。把strs[0]作为基准，依次与其他元素比较开头的元素，比较到不等的就截断-->
                截断的时候要注意是不是 执行完循环自动跳出，是的话就是包含最后一次比较的元素，否则就是在j处截断。
        :param strs:
        :return:
        """
        lengths = [len(s) for s in strs]
        if min(lengths) == 0:
            return ''
        ans = strs[0]
        for s in strs[1:]:
            flag_break = False
            for j in range(min(len(ans), len(s))):
                if ans[j] != s[j]:
                    flag_break = True
                    break
            if flag_break:
                ans = ans[:j]
            else:
                ans = ans[:j + 1]
        return ans

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        """
        15.三数之和
        中等 2022.09.23
        题解：set集合重复元素会自动过滤
        :param nums:
        :return:
        """
        nums.sort()
        res = []
        for i in range(len(nums) - 2):
            if nums[i] > 0:
                break
            left, right = i + 1, len(nums) - 1
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

    def threeSumClosest(self, nums: List[int], target: int) -> int:
        """
        16.最接近的三数之和
        中等 2022.09.24
        题解：脑袋瓦特+1，好长时间才做出来，且看了答案。因为nums长度是>3的，所以默认把前三个数的和det开始，后面for while配合，
                以当前三数和sum与target的绝对值 和 det与target的绝对值，变小了就更新det，最终返回det。不要想的太复杂嘛...刷题太少啦
        :param nums:
        :param target:
        :return:
        """
        nums.sort()
        det = nums[0] + nums[1] + nums[2]
        for i in range(len(nums) - 2):
            left, right = i + 1, len(nums) - 1
            while left < right:
                sum = nums[i] + nums[left] + nums[right]
                det = sum if abs(sum - target) < abs(det - target) else det
                if sum == target:
                    return sum
                elif sum < target:
                    left += 1
                else:
                    right -= 1
        return det

    def letterCombinations(self, digits: str) -> List[str]:
        """
        17.电话号码的字母组合
        中等 2022.09.24
        题解：输入为0位、1位时单独处理；>=2位时，第一位的字母拿出来，依次与后面每一位对应的字母进行拼接
        :param digits:
        :return:
        """
        dig_let = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}
        if len(digits) == 0:
            return []
        elif len(digits) == 1:
            return list(dig_let[digits])
        # 处理>=2位的
        res = list(dig_let[digits[0]])
        for key in digits[1:]:
            val = list(dig_let[key])
            tmp = []
            # 其实主要的就是val与res依次拼接
            for v in val:
                for r in res:
                    tmp.append(r + v)
            res = tmp
        return res

    def foursum(self, nums: List[int], target: int) -> List[List[int]]:
        """
        18.四数之和
        中等 2022.09.24
        题解：https://leetcode.cn/problems/4sum/solution/si-shu-zhi-he-by-leetcode-solution/
        :param nums:
        :param target:
        :return:
        """
        if len(nums) < 4:
            return []

        nums.sort()  # 因为示例上输出似乎升序
        res = set()  # 里面元素不能用 {}--set集合！！！可以忽略顺序，但也会自动过滤重复元素！！！
        for i in range(len(nums) - 3):
            for j in range(i + 1, len(nums) - 2):  # 这里仅考虑三个数相加就可以了
                left, right = j + 1, len(nums) - 1
                while left < right:
                    sum = nums[i] + nums[j] + nums[left] + nums[right]
                    if sum == target:
                        # if {nums[i], nums[j], nums[left], nums[right]} not in res:
                        res.add((nums[i], nums[j], nums[left], nums[right]))
                        left += 1  # 再让left right动起来，可能还有别的值符合要求呢
                    elif sum < target:
                        left += 1
                    else:
                        right -= 1
        return [list(t) for t in res]

    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        """
        19.删除链表的倒数第N个结点
        中等 2022.09.25
        题解：推算成正数第几个，似乎比较简单。正数第length-n个，前一个节点是第length-n-1个。需要特殊处理的是：如果length-n-1<0了代表
                要删除的是开始的第一个节点，直接返回head.next即可。
        :param head:
        :param n:
        :return:
        """
        # 题目说节点数目>=1
        length = 0  # 链表总长度
        cur = head
        while cur:
            length += 1
            cur = cur.next
        # 即正数第length-n个
        cur = head
        if length - n - 1 < 0:  # 如果即将删除的前一个节点出了合法范围
            return head.next
        for _ in range(length - n - 1):  # 找到即将删除的前一个节点
            cur = cur.next
        cur.next = cur.next.next
        return head

    def isValid(self, s: str) -> bool:
        """
        20.有效的括号
        简单 2022.09.25
        题解：方法一：s[0]和stack[-1]依次比较，相同则低效，否则入栈
                方法二：创意--将左右括号设计成key val的形式；遇到左括号--进栈、右括号--抵消，否则--提前return False，不用非得到最后才得出结果
        :param s:
        :return:
        """
        # s = list(s)
        # stack = []  # 用列表实现栈，官网的示例有这种
        # while s:
        #     if len(stack) and (stack[-1] is '(' and s[0] is ')' or
        #                             stack[-1] is '{' and s[0] is '}' or
        #                             stack[-1] is '[' and s[0] is ']'):
        #         stack.pop()
        #         s.pop(0)
        #     else:
        #         stack.append(s.pop(0))
        # return True if not len(stack) else False

        # leetcode解法 这个效率要高一些，提前return False
        dic = {'(': ')', '[': ']', '{': '}'}
        stack = []
        for c in s:
            if c in dic:  # 如果是左括号，入栈
                stack.append(c)
            elif len(stack) and c == dic[stack[-1]]:  # 先判断stack空不空，很重要；如果是右括号，出栈
                stack.pop()
            else:
                return False
        return True if not len(stack) else False

    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        """
        21.合并两个有序链表
        简单 2022.09.25
        :param list1:
        :param list2:
        :return:
        """
        # 方法一：新建一个链表返回
        # 题解：ListNode随存随建，不要提前建好，否则结果可能有一个多余的
        # res = self.ListNode(-1)     # 头结点
        # head = res
        # cur1, cur2 = list1, list2
        # while cur1 and cur2:
        #     if cur1.val < cur2.val:
        #         res.next = self.ListNode(cur1.val)
        #         res = res.next
        #         cur1 = cur1.next
        #     else:
        #         res.next = self.ListNode(cur2.val)
        #         res = res.next
        #         cur2 = cur2.next
        #
        # while cur1:
        #     res.next = self.ListNode(cur1.val)
        #     res = res.next
        #     cur1 = cur1.next
        # while cur2:
        #     res.next = self.ListNode(cur2.val)
        #     res = res.next
        #     cur2 = cur2.next
        # return head.next

        # 方法二：尝试本地操作
        # 题解：不如法一更简单。将list2的节点逐渐加到list1上--通过新建节点的方式--不算是本地计算
        if list1 and not list2:
            return list1
        elif not list1 and list2:
            return list2

        pre1 = self.ListNode(-1)
        head = pre1
        pre1.next = list1
        cur1, cur2 = list1, list2
        while cur1 and cur2:
            while cur1 and cur2 and cur1.val < cur2.val:  # 不要对空值取val
                pre1 = cur1
                cur1 = cur1.next
            pre1.next = self.ListNode(cur2.val)
            pre1.next.next = cur1
            pre1 = pre1.next

            cur2 = cur2.next
        if cur2:
            pre1.next = cur2
        return head.next

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

        # 再写一遍
        if n == 0:
            return []
        res_n = [[None], ['()']]  # 里面每个元素是n=1,2,3,4...时的情况
        for i in range(2, n + 1):
            tmp_i = []
            for j in range(i):
                list_p = res_n[j]
                list_q = res_n[i - 1 - j]
                for k1 in list_p:
                    for k2 in list_q:
                        k1 = "" if not k1 else k1
                        k2 = "" if not k2 else k2
                        ele = '(' + k1 + ')' + k2
                        tmp_i.append(ele)
            res_n.append(tmp_i)
        return res_n[n]

    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        24.两两交换链表中的节点
        中等 2022.09.27
        题解：自己想的感觉很复杂
                1.先考虑0和1个节点的情况
                2.定义pre cur next, res--res基本固定是第二个节点(除了0或1个节点)
                3.while cur and cur.next 是否处理当前的两个节点
                    3.1如果False即处理完了所有，处理下pre指向即可，否则会死循环
                    3.2如果True则处理新的两个节点，要注意的是pre的处理、next的指向
        题解：LeetCode思路，每次进栈两个，也能完成两两交换的目的。
                1.先考虑0/1个节点的情况
                2.cur:指向当前要处理的节点[1/2个，因为不一定每次都是两个，万一整个链表是奇数个呢]
                  pre:指向上次处理的两个节点的最后一个
                  stack:栈，每次只存2个节点，出栈的顺序刚好完成两两交换

                  节点入栈后就更新cur，指向下1/2个要处理的节点
                  第一次出栈用head承接，之后用pre连接
                3.while结束后要单独处理下pre指向——两种情况，剩下0/1个节点。
                  可能只剩下1个节点了，那么cur正指着；可能没节点了，pre.next指向None
        :param head:
        :return:
        """
        # if head is None:
        #     return None
        # elif head.next is None:
        #     return head
        #
        # # 处理>=2个节点的条件
        # cur = head
        # pre_cur = None      # 上一个cur所指的位置
        # next_cur = cur.next.next    # 下一个cur所指的位置
        # res = cur.next  # 返回的位置
        # while cur and cur.next:
        #     cur.next.next = cur
        #     if pre_cur:
        #         pre_cur.next = cur.next
        #     pre_cur = cur
        #     cur = next_cur if cur != next_cur else None
        #     if next_cur and next_cur.next and next_cur.next.next:    # 截断判断
        #         next_cur = next_cur.next.next
        #
        #     # else:
        #     #     break
        # pre_cur.next = cur if cur and cur.next is None else None
        # return res

        # leetcode思路，利用stack
        if head is None or head.next is None:
            return head

        cur = head
        pre = None  # 指向上次处理两节点的最后一个，便于连接到下一个两个节点
        stack = [cur, cur.next]
        cur = cur.next.next if cur.next.next else None  # 入栈后及时更新cur
        head = stack.pop()
        head.next = stack.pop()
        pre = head.next  # 出栈后，pre及时指向处理后两节点的最后一个
        while cur and cur.next:
            stack.append(cur)
            stack.append(cur.next)
            cur = cur.next.next if cur.next.next else None
            pre.next = stack.pop()
            pre.next.next = stack.pop()
            pre = pre.next.next
        pre.next = cur if cur else None
        return head

    def removeDuplicates(self, nums: List[int]) -> int:
        """
        26.删除有序数组中的重复项
        简单 2022.09.28
        :param nums:
        :return:
        """
        # 直接set去重，可是题目要求本地操作
        # nums = set(nums)
        # return list(nums)

        # 想到一种新方法，还是基于遍历——见到不同的元素往前面放 通过！
        last = 0  # 目前结果最后的元素
        for i in range(1, len(nums)):
            if nums[i] != nums[last]:
                nums[last + 1] = nums[i]
                last += 1
        return len(nums[:last + 1])

    def removeElement(self, nums: List[int], val: int) -> int:
        """
        27.移除元素
        简单 2022.09.28
        题解：参考LeetCode。遍历一次，定住一个位置last，用来放!=val的值，遍历一遍后就得到最后的结果了。比自己想的要简单很多！
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

    def nextPermutation(self, nums: List[int]) -> None:
        """
        31.下一个排列
        中等 2022.10.12
        题解：https://leetcode.cn/problems/next-permutation/solution/xia-yi-ge-pai-lie-suan-fa-xiang-jie-si-lu-tui-dao-/
            特殊处理1or2位数字。从右开始寻找相邻升序(因为降序代表最大了)，索引i,j，将i与后面大于nums[i]的最小的数nums[k]交换，
            然后将[j, end]升序，完成撒花~
        :param nums:
        :return:
        """
        if len(nums) == 1:
            return
        elif len(nums) == 2:
            nums.reverse()
            return

        i, j = len(nums) - 2, len(nums) - 1
        while nums[i] >= nums[j]:  # 寻找相邻升序
            if i - 1 < 0:  # 没找着，说明已经最大，则返回最小
                nums.sort()
                return
            i -= 1
            j -= 1
        for k in range(len(nums) - 1, j - 1, -1):  # 因为[j, end)为非升序，从右找能找到大于nums[i]的最小的数
            if nums[i] < nums[k]:
                nums[i], nums[k] = nums[k], nums[i]
                tmp = nums[j:]
                tmp.sort()
                nums[j:] = tmp
                break

    def binsearch(self, start, nums, target):
        """ 测试二分查找 """
        l, r = 0, len(nums) - 1
        while l <= r:
            m = (l + r) // 2
            if nums[m] == target:
                return m + start
            elif nums[m] < target:
                l = m + 1
            else:
                r = m - 1
        return -1

    def search(self, nums: List[int], target: int) -> int:
        """
        33.搜索旋转数组排序
        中等 2022.10.16
        题解：不知道如何处理时间复杂度要求o(logn)——从今天开始，再自己逞能只看题解就做题，我就是傻逼！
            https://leetcode.cn/problems/search-in-rotated-sorted-array/solution/sou-suo-xuan-zhuan-pai-xu-shu-zu-by-leetcode-solut/
            基于二分查找，从mid划分肯定有一边是非降序(考虑单元素时候，叫升序似乎不妥)，看target是否在此区间，否则在另一区间
            时间复杂度即二分查找，o(logn)
            空间复杂度o(1)
        :param nums:
        :param target:
        :return:
        """
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = (l + r) // 2
            if nums[mid] == target:
                return mid
            elif nums[l] <= nums[mid]:  # 左段只可能是升序 or 单元素
                if nums[l] <= target < nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1
            else:
                if nums[mid] < target <= nums[len(nums) - 1]:
                    l = mid + 1
                else:
                    r = mid - 1
        return -1

    def searchRange(self, nums: List[int], target: int) -> List[int]:
        """
        34. 在排序数组中查找元素的第一个和最后一个位置
        中等 2022.10.17
        题解：一看有序且时间复杂度o(logn)，想到上一题，二分查找
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
        while left >= 0 and nums[left] == nums[pos]:
            left -= 1
        left += 1

        right = pos + 1
        while right <= len(nums) - 1 and nums[right] == nums[pos]:
            right += 1
        right -= 1
        return [left, right]

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        """
        39.组合总和
        中等 2022.10.18
        题解：答案，回溯(不大懂)、递归
            https://leetcode.cn/problems/combination-sum/solution/hui-su-suan-fa-jian-zhi-python-dai-ma-java-dai-m-2/
        :param candidates:
        :param target:
        :return:
        """
        def dfs(candidates, begin, size, path, res, target):
            if target < 0:
                return
            if target == 0:
                res.append(path)
                return

            for ind in range(begin, size):
                dfs(candidates, ind, size, path + [candidates[ind]], res, target - candidates[ind])

        begin, size = 0, len(candidates)
        path, res = [], []
        dfs(candidates, begin, size, path, res, target)
        return res

    def permute(self, nums: List[int]) -> List[List[int]]:
        """
        46.全排列
        中等 2022.10.20
        题解:使用上题思路没有解出来。依然是回溯，不大理解。
            https://leetcode.cn/problems/permutations/solution/hui-su-suan-fa-python-dai-ma-java-dai-ma-by-liweiw/
        :param nums:
        :return:
        """
        def dfs(nums, size, depth, path, used, res):
            if depth == size:
                res.append(path[:])
                return

            for i in range(size):
                if not used[i]:
                    path.append(nums[i])
                    used[i] = True
                    dfs(nums, size, depth + 1, path, used, res)
                    used[i] = False
                    path.pop()

        size = len(nums)
        if not size:
            return []
        path, res = [], []
        used = [False for _ in range(size)]
        dfs(nums, size, 0, path, used, res)
        return res

    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        """
        47.全排列II
        中等 2022.10.23
        题解：
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
                    path.append(nums[i])
                    used[i] = True

                    dfs(nums, depth + 1, size, path, res, used)

                    path.pop()
                    used[i] = False

        size = len(nums)
        if not size:
            return []
        path, res = [], []
        used = [False for _ in range(size)]
        dfs(nums, 0, size, path, res, used)
        return res

    def rotate(self, matrix: List[List[int]]) -> None:
        """
        48.旋转图像
        中等 2022.10.24
        题解：旋转前后元素的对应关系
            https://leetcode.cn/problems/rotate-image/solution/xuan-zhuan-tu-xiang-by-leetcode-solution-vu3m/
        :param matrix:
        :return:
        """
        n = len(matrix)
        res = [[0] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):  # 原矩阵的行
            for j in range(n):  # 原矩阵的列
                res[j][n - 1 - i] = matrix[i][j]
        matrix[:] = res

    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        """
        49.字母异位词分组
        中等 2022.10.26
        题解：
            https://leetcode.cn/problems/group-anagrams/solution/zi-mu-yi-wei-ci-fen-zu-by-leetcode-solut-gyoc/
        :param strs:
        :return:
        """
        # 题解：
        default_mp = collections.defaultdict(list)
        for s in strs:
            default_mp[''.join(sorted(s))].append(s)
        return list(default_mp.values())


        # 自己写对了，但是很耗时。注意join、filter用法
        # dic = {}
        # key_start = 0
        # for s in strs:
        #     s = list(s)
        #     s.sort()
        #     s = ''.join(s)
        #
        #     if s not in dic.keys():
        #         dic[s] = key_start
        #         key_start += 1
        # res = [[] for _ in range(len(strs))]
        # for s in strs:
        #     old_s = s[:]
        #     s = list(s)
        #     s.sort()
        #     s = ''.join(s)
        #     res[dic[s]].append(old_s)
        #
        # def f(x):
        #     if len(x):
        #         return x
        # return [*filter(f, res)]

    def maxSubArray(self, nums: List[int]) -> int:
        """
        53.最大子数组和
        中等 2022.10.31
        题解：直接看代码就行
            https://leetcode.cn/problems/maximum-subarray/solution/dong-tai-gui-hua-fen-zhi-fa-python-dai-ma-java-dai/
        :param nums:
        :return:
        """
        sum_ind = [0 for _ in range(len(nums))]
        sum_ind[0] = nums[0]
        for i in range(1, len(nums)):
            if sum_ind[i - 1] > 0:
                sum_ind[i] = sum_ind[i - 1] + nums[i]
            else:
                sum_ind[i] = nums[i]
        return max(sum_ind)

    def canJump(self, nums: List[int]) -> bool:
        """
        55.跳跃游戏
        中等 2022.11.09
        题解：https://leetcode.cn/problems/jump-game/solution/pythonji-bai-97kan-bu-dong-ni-chui-wo-by-mo-lan-4/
            从索引0开始，使用enumerate()遍历nums，如果max_i可以到达当前i 且能跳到更远的地方 i+nums[i]>max_i，就更新max_i，最后与数组长度比较能不能到达结尾
        :param nums:
        :return:
        """
        max_i = 0
        for i, jump in enumerate(nums):
            if i <= max_i < i + jump:
                max_i = i + jump
        return max_i >= len(nums) - 1


if __name__ == '__main__':
    sl = Solution()

    # nums = [2, 7, 11, 15]
    # sl.twoSum(nums, 9)

    # l1 = sl.ListNode(2)
    # l1.next = sl.ListNode(4)
    # l1.next.next = sl.ListNode(3)
    #
    # l2 = sl.ListNode(5)
    # l2.next = sl.ListNode(6)
    # l2.next.next = sl.ListNode(4)
    #
    # head = sl.addTwoNumbers(l1, l2)
    # res = ''
    # while head:
    #     res = str(head.val) + res
    #     head = head.next
    # print(f'res={res}')

    # sl.lengthOfLongestSubstring("pwwkew")

    # print(sl.longestPalindrome("abcbacbd"))
    # print(sl.convert('PAYPALISHIRING', 4))

    # print(sl.reverse(123))

    # print(sl.myAtoi("words and -987"))

    # print(sl.isPalindrome(101))

    # height = [1, 1]
    # print(sl.maxArea(height))
    # print(sl.intToRoman(1994))
    # print(sl.romanToInt('MCMXCIV'))
    # strs = ["ab", "a"]
    # print(sl.longestCommonPrefix(strs))

    # a = {1, 2, 3}
    # b = {3, 2, 3}
    # print(b)
    # print(a == b)
    # a = {(1, 2, 3), (2, 1, 3)}
    # print(a)

    # nums = [-1,0,1,2,-1,-4]
    # print(sl.threeSum(nums))
    #
    # a = [1, 2, 3]
    # b = [3, 2, 1]
    # print(a == b)

    # nums = [-1,2,1,-4]
    # target = 1
    # print(sl.threeSumClosest(nums, target))

    # print('a' + 'b')
    # print(sl.letterCombinations('23'))

    # res = [{1, 2, 3}, {3, 2, 1}]    # set可以忽略顺序
    # print({2, 1, 3} in res)
    # nums = [1,0,-1,0,-2,2]
    # target = 0
    # print(sl.foursum(nums, target))

    # nums = list(range(1, 6))
    # head = sl.ListNode(nums.pop(0))
    # tmp = head
    # while nums:
    #     tmp.next = sl.ListNode(nums.pop(0))
    #     tmp = tmp.next
    #
    # res = sl.removeNthFromEnd(head, 1)
    # while res:
    #     print(res.val)
    #     res = res.next

    # s = '()[]{}]'
    # print(sl.isValid(s))

    # nums1 = [1,2,4]
    # head1 = sl.ListNode(nums1.pop(0))
    # tmp = head1
    # while nums1:
    #     tmp.next = sl.ListNode(nums1.pop(0))
    #     tmp = tmp.next
    #
    # nums2 = [1,3,4]
    # head2 = sl.ListNode(nums2.pop(0))
    # tmp = head2
    # while nums2:
    #     tmp.next = sl.ListNode(nums2.pop(0))
    #     tmp = tmp.next
    #
    # res = sl.mergeTwoLists(head1, head2)
    # while res:
    #     print(res.val)
    #     res = res.next

    # print(sl.generateParenthesis(3))

    # nums = [1]
    # head = sl.ListNode(nums.pop(0))
    # tmp = head
    # while nums:
    #     tmp.next = sl.ListNode(nums.pop(0))
    #     tmp = tmp.next
    # # while head:
    # #     print(head.val)
    # #     head = head.next
    # res = sl.swapPairs(head)
    # while res:
    #     print(res.val)
    #     res = res.next

    # nums = [0,0, 2]
    # print(sl.removeDuplicates(nums))

    # nums = [0,1,2,2,3,0,4,2]
    # val = 2
    # print(sl.removeElement(nums, val))

    # nums = [2, 2, 0, 4, 3, 1]
    # sl.nextPermutation(nums)
    # print(nums)

    # nums = list(range(10))
    # tmp = nums[7:]
    # tmp.sort(reverse=True)
    # nums[7:] = tmp
    # print(nums)

    # nums = [3, 1]
    # print(sl.search(nums, 1))

    # nums = []
    # target = 6
    # print(sl.searchRange(nums, target))

    # candidates = [2, 3, 6, 7]
    # target = 7
    # print(sl.combinationSum(candidates, target))

    # nums = [1, 2, 3]
    # print(sl.permute(nums))

    # nums = [1, 2, 1]
    # print(sl.permuteUnique(nums))

    # matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
    # sl.rotate(matrix)
    # print(matrix)
    # def f(x):
    #     if len(x):
    #         return x
    # nums = [[1, 2, 3], [], [3, 4]]
    # print(*filter(f, nums))
    # strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
    # strs = ["a"]
    # print(sl.groupAnagrams(strs))

    # nums = [5,4,-1,7,8]
    # print(sl.maxSubArray(nums))

    nums = [5, 9, 3, 2, 1, 0, 2, 3, 3, 1, 0, 0]
    print(sl.canJump(nums))