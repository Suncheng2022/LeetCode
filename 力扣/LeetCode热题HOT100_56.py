from typing import List, Optional


# Definition for singly-linked list.
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


class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        """
        56.合并区间
        中等 2023.02.01
        题解：https://leetcode.cn/problems/merge-intervals/solution/he-bing-qu-jian-by-leetcode-solution/
            先按第一个元素升序排序，如果 [res为空] 或 [当前元素的第一个元素 大于 res最后一个元素的第二个元素]，
            则不重合，当前元素直接放进res，否则用较大元素更新res最后一个元素的第二个元素
        :param intervals:
        :return:
        """
        # intervals.sort()  # 先排序，这是很重要的一步
        intervals.sort(key=lambda x: x[0])
        res = []
        for interval in intervals:
            if not res or res[-1][-1] < interval[0]:  # 与当前interval不重合
                res.append(interval)
            else:
                res[-1][-1] = max(res[-1][-1], interval[-1])
        return res

        # 自己想的，比较费劲
        # if len(intervals) == 1:
        #     return intervals
        #
        # intervals.sort()
        # res = []
        # tmp = intervals[0]    # 临时存储合并的区间，并初始化
        # i = 1
        # while i < len(intervals):
        #     if tmp[-1] >= intervals[i][0]:
        #         tmp[-1] = max(tmp[-1], intervals[i][-1])
        #     else:
        #         res.append(tmp)
        #         tmp = intervals[i]
        #     i += 1
        # res.append(tmp)
        # return res

    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        """
        57.插入区间
        中等 2023.02.02
        题解：借鉴56题思路
        :param intervals:
        :param newInterval:
        :return:
        """
        # 直接借鉴56. 简直一样解法
        # intervals.append(newInterval)
        # intervals.sort(key=lambda x: x[0])
        #
        # res = []
        # for interval in intervals:
        #     if not res or res[-1][-1] < interval[0]:  # 说明不重合
        #         res.append(interval)
        #     else:
        #         res[-1][-1] = max(res[-1][-1], interval[-1])
        # return res

        # 看题解，感觉更复杂了
        # 已经排好序了，遍历.
        res = []
        if len(intervals) == 0:
            res.append(newInterval)
            return res

        flag = 0
        for interval in intervals:
            if interval[-1] < newInterval[0]:  # 不重叠
                res.append(interval)
            elif interval[-1] >= newInterval[0] and interval[0] <= newInterval[-1]:  # 有重叠，则更新newInterval
                newInterval[0], newInterval[1] = min(interval[0], newInterval[0]), max(interval[-1], newInterval[-1])
            else:
                if not flag:
                    flag = 1
                    res.append(newInterval)
                res.append(interval)
        if not flag:
            res.append(newInterval)
        return res

    def lengthOfLastWord(self, s: str) -> int:
        """
        58.最后一个单词的长度
        简单 2023.02.06
        :param s:
        :return:
        """
        return len(s.split()[-1])

    def generateMatrix(self, n: int) -> List[List[int]]:
        """
        59.螺旋矩阵II
        中等 2023.02.06
        题解：https://leetcode.cn/problems/spiral-matrix-ii/solution/spiral-matrix-ii-mo-ni-fa-she-ding-bian-jie-qing-x/
        :param n:
        :return:
        """
        res = [[-1 for _ in range(n)] for _ in range(n)]
        print(f'test init res:{res}')
        curNum, maxNum = 1, n * n
        left, right, top, bottom = 0, n - 1, 0, n - 1
        while curNum <= maxNum:
            for i in range(left, right + 1):
                res[top][i] = curNum
                curNum += 1
            top += 1
            for i in range(top, bottom + 1):
                res[i][right] = curNum
                curNum += 1
            right -= 1
            for i in range(right, left - 1, -1):
                res[bottom][i] = curNum
                curNum += 1
            bottom -= 1
            for i in range(bottom, top - 1, -1):
                res[i][left] = curNum
                curNum += 1
            left += 1
        return res

    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        """
        61.旋转链表
        中等 2023.02.07
        题解：方法一：自己想的，好理解
            方法二：官方题解，思路也算好懂，代码不好懂，效率比较高
                    https://leetcode.cn/problems/rotate-list/solution/xuan-zhuan-lian-biao-by-leetcode-solutio-woq1/
        :param head:
        :param k:
        :return:
        """
        # nums = []
        # while head:
        #     nums.append(head.val)
        #     head = head.next
        # if not nums:
        #     return None
        # k %= len(nums)      # k特别大时，显著缩短运行时间
        # while k:
        #     n = nums.pop()
        #     nums.insert(0, n)
        #     k -= 1
        # # 返回一个新建的链表
        # head = cur = ListNode()
        # while nums:
        #     cur.val = nums.pop(0)
        #     if nums:
        #         cur.next = ListNode()
        #         cur = cur.next
        # return head

        # 方法二：
        if k == 0 or not head or not head.next:
            return head

        length = 0  # 链表长度，有几个元素
        cur = head
        while cur:
            length += 1
            cur = cur.next
            if not cur.next:  # cur已经指向了最后一个节点，在此形成闭环，结束while循环
                length += 1
                cur.next = head
                break

        k %= length
        count = length - k  # 这里要注意，根据k推出head最终指向正数第几个节点
        while count:
            head = head.next
            cur = cur.next
            count -= 1
        cur.next = None
        return head

    def uniquePaths(self, m: int, n: int) -> int:
        """
        62.不同路径
        中等 2023.02.09 动态规划
        题解：https://leetcode.cn/problems/unique-paths/solution/dong-tai-gui-hua-by-powcai-2/
        :param m:
        :param n:
        :return:
        """
        dp = [[1] * n] + [[1] + [0] * (n - 1) for _ in range(1, m)]  # 第一行、第一列均为1，其余元素为0
        for row in range(1, m):
            for col in range(1, n):
                dp[row][col] = dp[row - 1][col] + dp[row][col - 1]
        return dp[-1][-1]

    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        """
        63.不同路径II
        中等 2023.02.09
        题解：自己做，麻烦了，将近一个小时，错误...
            参考官方答案解出：https://leetcode.cn/problems/unique-paths-ii/solution/bu-tong-lu-jing-ii-by-leetcode-solution-2/
            1.初始化：将f初始化为和obstacleGrid同形状的全0数组，f[0][0]设为1（这并不妨碍此处为障碍的情况），用于后面计算
            2.遍历obstacleGrid，遇到1则f置0；特殊处理行列索引为0的情况；其余情况按f[i][j] = f[i - 1][j] + f[i][j - 1]处理
            3.结束
        :param obstacleGrid:
        :return:
        """
        # 我觉得改改能行，以后再说
        f = [[0] * len(obstacleGrid[0]) for _ in range(len(obstacleGrid))]
        f[0][0] = 1
        for i in range(len(obstacleGrid)):
            for j in range(len(obstacleGrid[0])):
                if obstacleGrid[i][j]:  # 有障碍物
                    f[i][j] = 0
                else:
                    if i == 0 and j > 0:  # 在索引0行
                        f[i][j] = f[i][j - 1]
                    elif j == 0 and i > 0:  # 在索引0列
                        f[i][j] = f[i - 1][j]
                    elif i > 0 and j > 0:
                        f[i][j] = f[i - 1][j] + f[i][j - 1]
        return f[-1][-1]

    def minPathSum(self, grid: List[List[int]]) -> int:
        """
        64.最小路径和
        中等 2023.02.12
        题解：自己思路，找到所有路径、计算所有路径代价。tip：区分grid元素和计算代价的cost元素
            并且与自己思路不谋而合：https://leetcode.cn/problems/minimum-path-sum/solution/zui-xiao-lu-jing-he-dong-tai-gui-hua-gui-fan-liu-c/
        :param grid:
        :return:
        """
        m, n = len(grid), len(grid[0])  # 行、列
        cost = [[-1] * n for _ in range(m)]
        cost[0][0] = grid[0][0]
        for i in range(m):
            for j in range(n):
                if i == 0 and j > 0:    # 当前元素仅受左边元素影响
                    cost[i][j] = grid[i][j] + cost[i][j - 1]
                elif j == 0 and i > 0:  # 当前元素仅受上边元素影响
                    cost[i][j] = grid[i][j] + cost[i - 1][j]
                elif i > 0 and j > 0:
                    cost[i][j] = min(grid[i][j] + cost[i - 1][j], grid[i][j] + cost[i][j - 1])
        return cost[-1][-1]

    def plusOne(self, digits: List[int]) -> List[int]:
        """
        66.加一
        简单 2023.02.13
        :param digits:
        :return:
        """
        number = 0
        for i in digits:
            number = number * 10 + i
        number += 1
        digits.clear()
        while number:
            digits.insert(0, number % 10)
            number //= 10   # 注意 整除 即可
        return digits

    def addBinary(self, a: str, b: str) -> str:
        """
        67.二进制求和
        简单 2023.02.14
        题解：感觉这个题并不简单呀。
        :param a:
        :param b:
        :return:
        """
        a = int(a, 2)   # 将参数（接受的类型挺多，其中包括str）转换为2进制（当然可以选择别的进制）
        b = int(b, 2)
        return bin(a + b).replace('0b', '')     # bin返回参数的二进制表示，是str，去掉前面的'0b'即可

    def mySqrt(self, x: int) -> int:
        """
        69.x的平方根
        简单 2023.02.15
        题解：https://leetcode.cn/problems/sqrtx/solution/niu-dun-die-dai-fa-by-loafer/
                牛顿迭代法，比较暴力，但只需要给足够的循环次数就能逼近结果
        :param x:
        :return:
        """
        # 牛顿迭代法
        # pow_x = 4
        # for _ in range(50):
        #     pow_x = (pow_x + x / pow_x) / 2
        # return int(pow_x)

        # 二分查找
        l, r = 0, x
        while l <= r:
            m = (l + r) // 2
            if m ** 2 <= x:
                res = m
                l = m + 1
            else:
                r = m - 1
        return res

    def climbStairs(self, n: int) -> int:
        """
        70.爬楼梯
        简单 2023.02.16
        题解：方法一：费锲那波数列 https://leetcode.cn/problems/climbing-stairs/solution/pa-lou-ti-by-leetcode-solution/
        时间复杂度：O(n)
        空间复杂度：O(n)
        :param n:
        :return:
        """
        res = [1, 2]
        if n <= 2:
            return res[n - 1]

        for i in range(2, n):
            tmp = res[i - 1] + res[i - 2]   # f(x) = f(x - 1) + f(x - 2)
            res.append(tmp)
        return res[-1]

    def simplifyPath(self, path: str) -> str:
        """
        71.简化路径
        中等 2023.02.13
        题解：https://leetcode.cn/problems/simplify-path/solution/zhan-by-powcai/
            栈的思路，遇到'..'弹出最后一个元素
        :param path:
        :return:
        """
        stack = []
        for item in path.split('/'):
            if item == '..' and stack:
                stack.pop()
            elif item and item not in ['.', '..']:
                stack.append(item)
        return '/' + '/'.join(stack)

    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        73.矩阵置零
        中等 2023.02.14
        题解：自己的做法，先记录下元素0所在的行列索引，遍历其行列后设为0，感觉不太高效
            也可参考 https://leetcode.cn/problems/set-matrix-zeroes/solution/ju-zhen-zhi-ling-by-leetcode-solution-9ll7/
        Do not return anything, modify matrix in-place instead.
        """
        # 答案的方法二，感觉还不如方法一
        row0_zero = False   # 用来标记matrix的第0行/列是否原就有 0
        col0_zero = False
        rows, cols = len(matrix), len(matrix[0])    # 先记录matrix行列
        for col in range(cols):     # 第一行原来是否有0
            if matrix[0][col] == 0:
                row0_zero = True
                break
        for row in range(rows):     # 第一列原来是否有0
            if matrix[row][0] == 0:
                col0_zero = True
                break

        for row in range(1, rows):      # 索引0行/列就不用操心啦，前面已经考虑了
            for col in range(1, cols):
                if matrix[row][col] == 0:
                    matrix[row][0] = 0    # 标记元素所在行列有0
                    matrix[0][col] = 0
        for row in range(1, rows):      # 遍历除索引0行/列的所有元素
            for col in range(1, cols):
                if matrix[row][0] == 0 or matrix[0][col] == 0:
                    matrix[row][col] = 0
        if row0_zero:
            for col in range(cols):
                matrix[0][col] = 0
        if col0_zero:
            for row in range(rows):
                matrix[row][0] = 0

        # 竟然与官方答案的方法一不谋而合
        # zero_rows = set()
        # zero_cols = set()
        # for row in range(len(matrix)):
        #     for col in range(len(matrix[0])):
        #         if matrix[row][col] == 0:
        #             zero_rows.add(row)
        #             zero_cols.add(col)
        # for row in zero_rows:
        #     matrix[row][:] = [0 for _ in range(len(matrix[0]))]
        # for col in zero_cols:
        #     for r in range(len(matrix)):
        #         matrix[r][col] = 0

    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        """
        74.搜索二维矩阵
        中等 2023.02.20
        题解：首先想到了二分法
        :param matrix:
        :param target:
        :return:
        """
        # 自己，尝试二分法
        # step1.找到<=target的行
        first_elements = [row[0] for row in matrix]
        l1, r1 = 0, len(first_elements) - 1
        while l1 <= r1:
            m1 = (l1 + r1) // 2
            if first_elements[m1] == target:
                return True
            elif first_elements[m1] < target:
                l1 = m1 + 1
            elif first_elements[m1] > target:
                r1 = m1 - 1
        # step2.定位最终位置
        row_elements = matrix[r1]
        l2, r2 = 0, len(row_elements) - 1
        while l2 <= r2:
            m2 = (l2 + r2) // 2
            if row_elements[m2] == target:
                return True
            elif row_elements[m2] < target:
                l2 = m2 + 1
            elif row_elements[m2] > target:
                r2 = m2 - 1
        return False

    def sortColors(self, nums: List[int]) -> None:
        """
        75.颜色分类
        中等 2023.02.21
        题解：原地排序，首先想到了冒泡排序
        :param nums:
        :return:
        """
        for i in range(len(nums) - 1):
            for j in range(1, len(nums) - i):
                if nums[j - 1] > nums[j]:
                    nums[j - 1], nums[j] = nums[j], nums[j - 1]

    def combine(self, n: int, k: int) -> List[List[int]]:
        """
        77.组合
        中等 2023.02.22
        题解：回溯法 https://leetcode.cn/problems/combinations/solution/dai-ma-sui-xiang-lu-dai-ni-xue-tou-hui-s-0uql/
        :param n:
        :param k:
        :return:
        """
        path = []
        res = []

        def fun(n, k, startIdx):
            if len(path) == k:
                res.append(path[:])
                return
            for i in range(startIdx, n + 1):
                path.append(i)
                fun(n, k, i + 1)        # 此处第三个参数是 i+1，而不是startInd + 1
                path.pop()
        fun(n, k, 1)
        return res

        # path = []
        # res = []
        #
        # def fun(n, k, start_index):
        #     if len(path) == k:
        #         res.append(path[:])
        #         return
        #     for i in range(start_index, n + 1):
        #         path.append(i)
        #         fun(n, k, i + 1)
        #         path.pop()
        #
        # fun(n, k, 1)
        # return res

    def subsets(self, nums: List[int]) -> List[List[int]]:
        """
        78.子集
        中等 2023.02.26
        题解：《代码随想录》非常不错 https://programmercarl.com/0078.%E5%AD%90%E9%9B%86.html#c-%E4%BB%A3%E7%A0%81
        :param nums:
        :return:
        """
        path = []
        res = []

        def fun(nums, startIndex):
            res.append(path[:])
            if startIndex >= len(nums):
                return
            for i in range(startIndex, len(nums)):
                path.append(nums[i])
                fun(nums, i + 1)
                path.pop()

        fun(nums, 0)
        return res

    def exist(self, board: List[List[str]], word: str) -> bool:
        """
        79.单词搜索【放弃】
        中等 2023.02.27
        题解:回溯 https://leetcode.cn/problems/word-search/solution/shen-du-you-xian-sou-suo-yu-hui-su-xiang-jie-by-ja/
        :param board:
        :param word:
        :return:
        """
        def fun(i, j, word):
            if len(word) == 0:
                return True
            for direct in directs:
                cur_i, cur_j = i + direct[0], j + direct[1]
                if 0 <= cur_i < m and 0 <= cur_j < n and board[cur_i][cur_j] == word[0]:
                    if mark[cur_i][cur_j]:
                        continue
                    mark[cur_i][cur_j] = 1
                    if fun(cur_i, cur_j, word[1:]):
                        return True
                    mark[cur_i][cur_j] = 0
            return False

        if len(board) == 0:
            return False
        directs = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        m, n = len(board), len(board[0])
        mark = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if board[i][j] == word[0]:
                    mark[i][j] = 1
                    if fun(i, j, word[1:]):
                        return True
                    mark[i][j] = 0
        return False




        # def fun(i, j, word):
        #     if len(word) == 0:
        #         return True
        #     for direct in directs:
        #         cur_i = i + direct[0]
        #         cur_j = j + direct[1]
        #         if 0 <= cur_i < m and 0 <= cur_j < n and board[cur_i][cur_j] == word[0]:
        #             if mark[cur_i][cur_j]:
        #                 continue
        #             mark[cur_i][cur_j] = 1
        #             if fun(cur_i, cur_j, word[1:]): return True
        #             else: mark[cur_i][cur_j] = 0
        #     return False
        #
        # directs = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        # m, n = len(board), len(board[0])
        # mark = [[0] * n for _ in range(m)]
        # if m == 0:
        #     return False
        # for i in range(m):
        #     for j in range(n):
        #         if board[i][j] == word[0]:
        #             mark[i][j] = 1
        #             if fun(i, j, word[1:]): return True
        #             else: mark[i][j] = 0
        # return False

    def removeDuplicates(self, nums: List[int]) -> int:
        """
        80.删除有序数组中的重复项II
        中等 2023.03.01
        题解：https://leetcode.cn/problems/remove-duplicates-from-sorted-array-ii/solution/gong-shui-san-xie-guan-yu-shan-chu-you-x-glnq/
            还是题解的思路逻辑清晰。当前写入的索引为curIdx，当前处理的元素n用遍历获取，然后比较n与nums[curIdx - 2]是否相等.
        :param nums:
        :return:
        """
        def keepK(k):
            curIdx = 0
            for n in nums:
                if curIdx < k or n != nums[curIdx - k]:
                    nums[curIdx] = n
                    curIdx += 1
            return curIdx

        return keepK(2)

        # 自己想的还是比较复杂，记得看答案
        # if len(nums) <= 2:
        #     return len(nums)
        # curIdx = 0
        # counts = 0
        # tmp = nums[0]
        # offset = 0
        # for i, n in enumerate(nums):
        #     if n == tmp and counts < 2:
        #         counts += 1
        #         nums[curIdx] = n
        #         curIdx += 1
        #     elif n == tmp and counts >= 2:
        #         counts += 1
        #         offset += 1
        #         continue
        #     elif n != tmp:
        #         nums[curIdx] = n
        #         curIdx += 1
        #         tmp = n
        #         counts = 1
        # return len(nums[:-offset]) if offset else len(nums)

    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        """
        94.二叉树的中序遍历
        简单 2023.03.05
        题解：我觉得不如while循环好理解（见提交记录）
        :param root:
        :return:
        """
        res = []

        def fun(root):
            if root and root.left:
                fun(root.left)
            if root:
                res.append(root.val)
            if root and root.right:
                fun(root.right)
        fun(root)
        return res

    def numTrees(self, n: int) -> int:
        """
        96.不同的二叉搜索树
        中等 2023.03.06
        题解：https://leetcode.cn/problems/unique-binary-search-trees/solution/shou-hua-tu-jie-san-chong-xie-fa-dp-di-gui-ji-yi-h/
        :param n:
        :return:
        """
        dp = [0 for _ in range(n + 1)]     # i:索引  元素:用连着的i个数组成的二叉搜索树的数量
        dp[0] = 1   # 0个节点只能组成一种树：空树
        dp[1] = 1   # 1个节点只能组成一种树：1个节点的树
        for i in range(2, n + 1):   # 求dp[i]，即求i为根节点时二叉搜索树的数目
            for j in range(0, i):   # 根节点为i，则剩余节点为[0, i - 1]，由这些节点组成左右子树
                dp[i] += dp[j] * dp[i - 1 - j]
        return dp[n]

    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        """
        98.验证二叉搜索树
        中等 2023.03.07
        题解：https://leetcode.cn/problems/validate-binary-search-tree/solution/yan-zheng-er-cha-sou-suo-shu-by-leetcode-solution/
            按照中序遍历顺序访问节点，则访问节点一定大于前一个节点，否则不是二叉搜索树。
        通过while循环的方式实现中序遍历比较容易懂，比递归好理解得多。
        :param root:
        :return:
        """
        stack = []
        inorder = float('-inf')

        while stack or root:
            while root:
                stack.append(root)      # 循环添加左节点，弹出时能访问中间节点，同时能赋值为右节点继续来此循环添加
                root = root.left
            node = stack.pop()
            if node.val <= inorder:      # 中序遍历，若<=前一个数，则不是二叉搜索树
                return False
            inorder = node.val
            root = node.right
        return True

    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        """
        101.对称二叉树
        简单 2023.03.08
        题解：首先想到中序遍历对称则是对称二叉树，否则不是——没做出来...耗时很多
            看答案，队列实现https://leetcode.cn/problems/symmetric-tree/solution/dong-hua-yan-shi-101-dui-cheng-er-cha-shu-by-user7/
        :param root:
        :return:
        """
        if not root or not (root.left or root.right):   # 空树 和 只有根节点
            return True
        queue = [root.left, root.right]     # 若root只1个孩子节点也没事
        while queue:
            left = queue.pop()      # 感觉这里应该是pop(0)
            right = queue.pop()
            if not (left or right):     # 二者均空，继续
                continue
            if not (left and right) or left.val != right.val:    # 若二者值不等 或 二者其一为空；先判断是否有空，否则报错
                return False
            queue.extend([left.left, right.right, left.right, right.left])
        return True

    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        """
        102.二叉树的层序遍历
        中等 2023.03.09
        题解：能大致想到思路，先进先出；但如何标记层
            层序遍历，完美 https://leetcode.cn/problems/binary-tree-level-order-traversal/solution/die-dai-di-gui-duo-tu-yan-shi-102er-cha-shu-de-cen/
        :param root:
        :return:
        """
        if not root:    # 琢磨一下方法的实现，是不允许queue中出现空节点的，所以开始要判断一下root是否为空，不能直接把root放进queue
            return []
        queue = [root]  # 队列，FIFO
        res = []
        while queue:
            numEleLayer = len(queue)    # while一次处理一层，所以这里获取的是即将处理的一层所拥有的元素数量
            tmp = []
            for _ in range(numEleLayer):
                node = queue.pop(0)
                tmp.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(tmp)
        return res

        # 没有全通过，耗时30分钟
        # stack = [root]      # 先进先出
        # res = []
        # count = 0
        # nextLayer = 2 ** 0
        # while stack:
        #     p = stack.pop(0)
        #     count += 1
        #     if count % nextLayer == 0:
        #         res.append('-')
        #         nextLayer *= 2
        #     while (not p) and stack:
        #         p = stack.pop(0)
        #         count += 1
        #         if count % nextLayer == 0:
        #             res.append('-')
        #             nextLayer *= 2
        #     if not p:
        #         break
        #     res.append(p.val)
        #     stack.extend([p.left, p.right])
        # finalRes = []
        # tmp = []
        # for c in res[1:]:
        #     if c == '-':
        #         finalRes.append(tmp)
        #         tmp.clear()
        #         continue
        #     tmp.append(c)
        # return finalRes

    def maxDepth(self, root: Optional[TreeNode]) -> int:
        """
        104.二叉树的最大深度
        简单 2023.03.11
        题解：102.二叉树的层序遍历
        :param root:
        :return:
        """
        if not root:
            return 0
        queue = [root]
        res = []
        while queue:
            num_of_layer = len(queue)
            tmp = []
            for _ in range(num_of_layer):
                node = queue.pop(0)
                tmp.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(tmp)
        print(f'层序遍历:{res}')
        return len(res)

    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        """
        105.从前序与中序遍历序列构造二叉树
        中等 2023.03.13
        题解：https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/solution/cong-qian-xu-yu-zhong-xu-bian-li-xu-lie-gou-zao-9/
        :param preorder:
        :param inorder:
        :return:
        """
        val2ind_inorder = {val: i for i, val in enumerate(inorder)}

        def buildSubTree(preorder_left, preorder_right, inorder_left, inorder_right):
            """ 参数理解为buildTree参数的索引——均指示同一[子]树 """
            # 递归的停止条件
            if preorder_left > preorder_right:
                return None
            # 先找根节点在先序、中序遍历的索引
            preorderRoot = preorder_left
            inorderRoot = val2ind_inorder[preorder[preorder_left]]
            rootNode = TreeNode(preorder[preorder_left])    # 构造根节点
            size_LeftSubtree = inorderRoot - inorder_left   # 左子树的长度，则右子树的长度自然明了，但不必计算出来，那样反而麻烦了
            rootNode.left = buildSubTree(preorder_left + 1, preorder_left + size_LeftSubtree, inorder_left, inorderRoot - 1)    # 索引指示同一子树
            rootNode.right = buildSubTree(preorder_left + size_LeftSubtree + 1, preorder_right, inorderRoot + 1, inorder_right)
            return rootNode

        n = len(val2ind_inorder)
        return buildSubTree(0, n - 1, 0, n - 1)



        # index = {val: i for i, val in enumerate(inorder)}       # 以中序遍历建立哈希字典，随机访问加快
        #
        # def bulidRoot(preorder_left, preorder_right, inorder_left, inorder_right):
        #     """ 参数均为 要建立的子树不同遍历顺序的索引 """
        #     if preorder_left > preorder_right:
        #         return None
        #     preorder_root = preorder_left
        #     inorder_root = index[preorder[preorder_left]]
        #     size_leftsubtree = inorder_root - inorder_left
        #     root = TreeNode(preorder[preorder_left])
        #     root.left = bulidRoot(preorder_left + 1, preorder_left + size_leftsubtree, inorder_left, inorder_root - 1)     # 参数 为 刚建立的root的左子树在preorder中的索引...
        #     root.right = bulidRoot(preorder_left + size_leftsubtree + 1, preorder_right, inorder_root + 1, inorder_right)
        #     return root
        # n = len(index)
        # return bulidRoot(0, n - 1, 0, n - 1)

if __name__ == '__main__':
    sl = Solution()


    # root = [3, 9, 20, None, None, 15, 7]
    # print(sl.levelOrder(root))

    # print(sl.numTrees(1))

    # nums = [0,0,1,1,1,1,2,3,3]
    # print(sl.removeDuplicates(nums))

    # board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
    # word = "ABCB"
    # print(sl.exist(board, word))

    # nums = [0]
    # print(sl.subsets(nums))

    # print(sl.combine(1, 1))

    # nums = [2,0,2,1,1,0]
    # sl.sortColors(nums)
    # print(nums)

    # matrix = [[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]]
    # target = 3
    # # matrix = [[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]]
    # # target = 13
    # print(sl.searchMatrix(matrix, target))

    # print(sl.climbStairs(1))

    # print(sl.mySqrt(0))

    # matrix = [[1,1,1],[1,0,1],[1,1,1]]
    # sl.setZeroes(matrix)
    # print(matrix)

    # a = '1010'
    # b = '1011'
    # print(sl.addBinary(a, b))

    # path = "/home//foo/"
    # print(sl.simplifyPath(path))
    # digits = [0]
    # print(sl.plusOne(digits))

    # grid = [[1,2,3],[4,5,6]]
    # print(sl.minPathSum(grid))

    # obstacleGrid = [[0, 0], [0, 1]]
    # print(sl.uniquePathsWithObstacles(obstacleGrid))

    # head = [1, 2, 3, 4, 5]
    # p = cur = ListNode()
    # while head:
    #     cur.val = head.pop(0)
    #     if head:
    #         cur.next = ListNode()
    #         cur = cur.next
    # # while p:
    # #     print(p.val)
    # #     p = p.next
    # # exit()
    # k = 2
    # res = sl.rotateRight(p, k)
    # while res:
    #     print(res.val)
    #     res = res.next

    # print(sl.generateMatrix(4))

    # intervals = [[1, 5]]
    # newInterval = [4, 8]
    # print(sl.insert(intervals, newInterval))

    # intervals = [[1, 3], [2, 6], [8, 10], [15, 18]]
    # print(sl.merge(intervals))
