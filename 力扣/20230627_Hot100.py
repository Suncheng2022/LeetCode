import collections
from typing import List, Optional


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


from collections import defaultdict
class Node:
    """ 208.实现Trie(前缀树)，所需数据结构 """
    def __init__(self):
        self.children = defaultdict(Node)   # 每个节点有多个叉，相当于多叉树；本题处理单词，每个节点有26个可能的叉，即26叉树
        self.isword = False
class Trie:
    """
    208.实现Trie(前缀树)
    2023.07.08 中等
    题解：还不错的解析，耐心看一遍吧
        https://leetcode.cn/problems/implement-trie-prefix-tree/solutions/721050/fu-xue-ming-zhu-cong-er-cha-shu-shuo-qi-628gs/
    """
    def __init__(self):
        self.root = Node()      # Trie前缀树根节点不保存信息

    def insert(self, word: str) -> None:
        curr_node = self.root
        for c in word:
            curr_node = curr_node.children[c]       # curr_node最终会指向单词最后一个字母；因为children使用的是defaultdict，因此insert时没有子节点会新建
        curr_node.isword = True

    def search(self, word: str) -> bool:
        curr_node = self.root
        for c in word:
            curr_node = curr_node.children.get(c, None)
            if curr_node is None:
                return False
        return curr_node.isword     # curr_ndoe最终会指向单词的最后一个字母，虽然word所有字母都能依次找到，但trie不一定存储了word这个词

    def startsWith(self, prefix: str) -> bool:
        curr_node = self.root
        for c in prefix:
            curr_node = curr_node.children.get(c, None)
            if curr_node is None:
                return False
        return True             # 只要前缀存在即可，在前缀树中prefix不一定非得是词

class MinStack:
    """
    155.最小栈 中等
    2023.07.09
    题解：答案 巧妙的维护最小值
    """
    def __init__(self):
        self.stack = []
        self.mini_stack = []    # 栈底至当前元素区间的最小值元素

    def push(self, val: int) -> None:
        # 主要是如何维护最小值
        self.stack.append(val)
        if len(self.mini_stack) == 0:
            self.mini_stack.append(val)
        else:
            self.mini_stack.append(min(val, self.mini_stack[-1]))

    def pop(self) -> None:
        _ = self.stack.pop()
        _ = self.mini_stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.mini_stack[-1]


from collections import OrderedDict

class LRUCache:
    """
    146.LRU缓存
    2023.07.09 中等
    题解：OrderedDict 有move_to_end()方法
        使用自己方法试试
    """
    # 自己的方法 有点绕哦
    def __init__(self, capacity):
        self.capacity = capacity
        self.recent_lru = []        # set()虽然不包含重复元素，但失去了顺序的特性，因此还是使用列表
        self.lru = {}

    def get(self, key):
        if key in self.lru:
            self.recent_lru.remove(key)
            self.recent_lru.insert(-1, key)
            return self.lru[key]
        return -1

    def put(self, key, value):
        if key in self.lru:
            self.lru[key] = value
            self.recent_lru.remove(key)
            self.recent_lru.insert(-1, key)
        else:
            self.lru[key] = value
            self.recent_lru.append(key)

        if len(self.lru) > self.capacity:
            self.lru.pop(self.recent_lru[0])
            self.recent_lru.pop()



    # 答案 使用OrderedDict相当于自动维护
    # def __init__(self, capacity: int):
    #     self.capacity = capacity
    #     self.ord_cache = OrderedDict()
    #
    # def get(self, key: int) -> int:
    #     if key in self.ord_cache:
    #         self.ord_cache.move_to_end(key)        # 最近访问的放后面
    #         return self.ord_cache[key]
    #     return -1
    #
    # def put(self, key: int, value: int) -> None:
    #     self.ord_cache[key] = value
    #     self.ord_cache.move_to_end(key)
    #     if len(self.ord_cache) > self.capacity:
    #         self.ord_cache.popitem(last=False)


class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        """
        739.每日温度
        2023.06.27 中等
        题解：递减栈
        """
        stack = []      # 维护递减栈
        n = len(temperatures)
        res = [0] * n
        for i in range(n):
            if not stack or temperatures[i] <= stack[-1][-1]:
                stack.append([i, temperatures[i]])
            # 当temperatures[i] > stack[-1][-1]时，肯定是栈中元素第一次遇到较大元素
            while stack and temperatures[i] > stack[-1][-1]:
                ind, _ = stack.pop()
                res[ind] = i - ind      # 栈顶元素第一次遇见较大元素，计算间隔天数
            stack.append([i, temperatures[i]])
        return res

    def countSubstrings(self, s: str) -> int:
        """
        647.回文子串
        2023.06.27 中等
        题解：向两端探测，稍作修改，因为题目要求位置不同即算视作不同子串
        """
        res = set()     # set可去重
        for i in range(len(s)):
            res.add((i))        # 本身也是回文子串
            l, r = i - 1, i + 1
            while l >= 0 and s[l] == s[i]:      # 向左探测所有与s[i]相同的元素
                res.add((l, i))
                l -= 1
            while r < len(s) and s[r] == s[i]:  # 向右探测所有与s[i]相同的元素
                res.add((i, r))
                r += 1
            while l >= 0 and r < len(s):
                if s[l] == s[r]:
                    res.add((l, r - 1))
                    l -= 1
                    r += 1
                else:
                    break
        return res.__len__()

    def leastInterval(self, tasks: List[str], n: int) -> int:
        """
        621.任务调度器
        2023.06.27 中等
        题解：res = (max_freq - 1) * (n + 1) + num of max_freq
        """
        from collections import Counter

        task2counts = sorted(Counter(tasks).items(), key=lambda x: x[1], reverse=True)
        max_freq = task2counts[0][-1]
        num_max_freq = sum([True for _, counts in task2counts if counts == max_freq])
        # 最后加num_max_freq，因为还有自己，所以一共num_max_freq个
        res = (max_freq - 1) * (n + 1) + num_max_freq
        return res if res >= len(tasks) else len(tasks)

    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
        """
        617.合并二叉树
        2023.06.27 简单
        题解：回忆答案 递归
        """
        # 递归 前序遍历
        # def dfs(root1, root2):
        #     if not (root1 and root2):       # 至少1个空
        #         return root1 if root1 else root2
        #     root1.val += root2.val
        #     root1.left = dfs(root1.left, root2.left)
        #     root1.right = dfs(root1.right, root2.right)
        #     return root1
        #
        # return dfs(root1, root2)

        # 迭代 广度优先遍历
        # 广度优先需要辅助存储了
        if not (root1 and root2):   # 只要有1个空
            return root1 if root1 else root2
        queue = [[root1, root2]]
        while queue:
            node1, node2 = queue.pop(0)
            node1.val += node2.val
            if node1.left and node2.left:   # node1.left、node2.left都不空直接入队
                queue.append([node1.left, node2.left])
            elif not node1.left:            # node1.left空、node2.left不空，拷贝过去；node1.left不空、node2.left空，不用处理，反正最后返回root1
                node1.left = node2.left
            if node1.right and node2.right:     # node1.right、node2.right同理
                queue.append([node1.right, node2.right])
            elif not node1.right:
                node1.right = node2.right
        return root1        # 最终返回root1

    def findUnsortedSubarray(self, nums: List[int]) -> int:
        """
        581.最短无序连续子数组
        2023.06.27 中等
        题解：从左往右找 非升序 最大值，从右往左找 非升序 最小值
        """
        max_num, right = float('-inf'), -1
        min_num, left = float('inf'), -1
        n = len(nums)
        for i in range(n):
            if nums[i] < max_num:
                right = i       # 从左往右，遇到那段 非升序 就更新right；最终指向 非升序 的最右边元素
            else:
                max_num = nums[i]
            if nums[n - 1 - i] > min_num:
                left = n - 1 - i    # 从右往左，遇到那段 非升序 就更新right；最终指向 非升序 的最左边元素
            else:
                min_num = nums[n - 1 - i]
        return right - left + 1 if right != left else 0     # 可能不需要调整；注意返回的是 长度

    def subarraySum(self, nums: List[int], k: int) -> int:
        """
        560.和为K的子数组
        2023.06.27 中等
        题解：前缀和。当前的前缀和presum，前缀和presum-k出现的次数
        """
        from collections import defaultdict

        # 前缀和及其出现次数
        preSums = defaultdict(int)
        preSums[0] = 1
        res = 0
        presum = 0      # 当前的前缀和
        for i in range(len(nums)):
            presum += nums[i]       # 遍历，当前的前缀和
            # 是否有前缀和为presum-k的，并累加出现次数；
            # 【当前的前缀和presum是从头一个元素一个元素遍历累加得来的，肯定会在某个时刻超过k；我们不断寻找前面是否出现过大小为presum-k的前缀和，二者的距离就是k】
            res += preSums[presum - k]
            preSums[presum] += 1    # 当前的前缀和出现次数+1
        return res

    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        """
        543.二叉树的直径
        2023.06.27 中等
        题解：递归，求树的高度；递归过程不断更新 路径
        """
        def depth(node):
            """ 计算根节点为node时数的高度 """
            if not node:
                return 0
            l = depth(node.left)
            r = depth(node.right)
            nonlocal res
            res = max(res, l + r)       # 更新 路径，不是计算树的高度，所以不用 +1
            return max(l, r) + 1       # 树的高度要加根节点自己，即+1

        res = 0
        depth(root)
        return res

    def convertBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """
        538.把二叉搜索树转换为累加树
        2023.06.28 中等
        题解：能想到 右根左 的递归遍历顺序
        """

        # *序遍历似乎都是：开始判断一下--递归终止条件；在遍历根节点时操作
        def dfs(node):
            if not node:
                return 0
            dfs(node.right)
            nonlocal total
            total += node.val
            node.val = total
            dfs(node.left)

        total = 0
        dfs(root)
        return root

    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        """
        494.目标和
        2023.06.28 中等
        题解：数组元素均非负，数组和sum，带负号元素和绝对值neg，正号元素和绝对值sum-neg，因此有sum-neg-neg=target-->(sum-target)//2=neg，求能组成neg的方案数
        """
        sum_nums = sum(nums)
        if sum_nums < target or (sum_nums - target) % 2:    # 如果数组和小于target、数组和-target不是偶数，那么没有方案
            return 0
        neg = (sum_nums - target) // 2
        n = len(nums)
        dp = [[0] * (neg + 1) for _ in range(n + 1)]    # dp[i][j]: nums前i个数能选出和为j的方案数
        dp[0][0] = 1    # dp[0][0]=1, dp[0][>=1]均为0
        for i in range(1, n + 1):       # 【0行已初始化，这里不再计算】！！！
            for j in range(neg + 1):
                if nums[i - 1] > j:    # 不能选nums[i]
                    dp[i][j] = dp[i - 1][j]     # i-1表达了不选nums[i]
                else:
                    dp[i][j] = dp[i - 1][j] + dp[i - 1][j - nums[i - 1]]    # j-nums[i]表达了必选nums[i]
        return dp[-1][-1]

    def hammingDistance(self, x: int, y: int) -> int:
        """
        461.汉明距离
        2023.06.28 简单
        """
        tmp = x ^ y
        res = 0
        while tmp:
            if tmp & 1:
                res += 1
            tmp >>= 1
        return res

    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        """
        448.找到所有数组中消失的数字
        2023.06.29 简单
        题解：自己想的可能简单了
        """
        # 答案 重写 元素值映射到索引
        for v in nums:      # 元素v对应的索引处的值，代表v是否出现过
            if nums[abs(v) - 1] > 0:
                nums[abs(v) - 1] *= -1
        return [i + 1 for i, v in enumerate(nums) if v > 0]     # 最终，v>0代表v所在索引位置没有置负，即索引位置代表的元素没有出现过；注意+1

        # # 答案 直接使用原数组标记是否出现过
        # for n in nums:
        #     if nums[abs(n) - 1] > 0:
        #         nums[abs(n) - 1] *= -1       # 元素值映射到索引，要-1，置负；nums[abs[n]-1] 数组索引范围[0,n-1]，元素介于1~n
        # return [i + 1 for i, n in enumerate(nums) if n > 0]


        # n = len(nums)
        # res = [False] * (n + 1)
        # for k in nums:
        #     res[k] = True
        # return [i for i in range(1, n + 1) if not res[i]]

    def findAnagrams(self, s: str, p: str) -> List[int]:
        """
        438.找到字符串中所有字母异位词
        2023.06.29 中等
        题解：从s上滑窗连续m个元素，统计出现次数是否等于p的元素出现次数；s滑动时，每滑动一位要删除一位旧元素
        """
        n = len(s)
        m = len(p)
        if n < m:
            return []
        res = []
        # 统计s、p字符串连续m个位置字母出现次数
        s_cnt = [0] * 26
        p_cnt = [0] * 26
        for i in range(m):      # 统计第1个连续m个元素出现次数
            s_cnt[ord(s[i]) - ord('a')] += 1
            p_cnt[ord(p[i]) - ord('a')] += 1
        if s_cnt == p_cnt:
            res.append(0)       # 以索引0起始的m位元素与p是异位词
        # 滑动窗口遍历剩下的元素，出一个旧元素、进一个新元素
        for i in range(m, n):
            s_cnt[ord(s[i - m]) - ord('a')] -= 1    # 这不好懂，滑动窗口一共m个位置，此时遍历到索引i，那索引i-m就出去了--即索引i-m所指元素出现次数减1
            s_cnt[ord(s[i]) - ord('a')] += 1        # 滑动窗口遍历到索引i，即索引i所指元素出现次数加1
            if s_cnt == p_cnt:      # 每向右滑一次都要检查是否与p异位词
                res.append(i - m + 1)       # 遍历到索引i时，索引i-m刚好出去，所以起始元素是i-m+1
        return res

    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        """
        437.路径总和III
        2023.06.30 中等
        题解：前缀和 回溯 递归
        """
        def recursionPathSum(node, prefixSumMap, currSum, target):
            if not node:
                return 0
            res = 0
            currSum += node.val
            res += prefixSumMap.get(currSum - target, 0)
            prefixSumMap[currSum] = prefixSumMap.get(currSum, 0) + 1

            res += recursionPathSum(node.left, prefixSumMap, currSum, target)
            res += recursionPathSum(node.right, prefixSumMap, currSum, target)
            # 回溯就要去掉进来的
            prefixSumMap[currSum] -= 1
            return res

        prefixSumMap = {0: 1}
        return recursionPathSum(root, prefixSumMap, 0, targetSum)

    def canPartition(self, nums: List[int]) -> bool:
        """
        416.分割等和子集
        2023.06.30 中等
        题解：动态规划 dp[i][j]：从nums索引0~i选取和为j的元素
        """
        n = len(nums)
        if n < 2:
            return False
        maxNum = max(nums)
        total = sum(nums)
        if total % 2:
            return False
        target = total // 2     # 选取元素和为target，数组元素和的一半
        if maxNum > target:
            return False
        # 进入正题，动态规划
        # 构造dp数组，dp[0][j>0]均为False，dp[i][0]=True
        dp = [[True] + [False] * target for _ in range(n)]     # n行 target+1列
        for i in range(1, n):               # 因为dp[0][j>0]均为False，所以遍历从索引1开始
            for j in range(1, target + 1):  # 因为dp[i][0]均为True，所以遍历从索引1开始
                if nums[i] > j:     # 当前遍历元素nums[i]>j，选了就大于j，所以不选
                    dp[i][j] = dp[i - 1][j]
                else:               # 当前遍历元素nums[i]<=j，可选可不选
                    dp[i][j] = dp[i - 1][j] or dp[i - 1][j - nums[i]]
        return dp[-1][-1]

    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        """
        406.根据身高重建队列
        2023.06.30 中等
        题解：对一个属性升序、另一个降序，基本放在了对的位置，然后遍历微调
        """
        people = sorted(people, key=lambda x: (-x[0], x[1]))
        res = []
        for p in people:
            if len(res) <= p[1]:
                res.append(p)
            else:
                res.insert(p[1], p)
        return res

    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        """
        399.除法求值
        2023.06.30 中等
        题解：
        """
        # 用已知构造 图，表达两两之间的乘除
        graph = {}
        for (x, y), v in zip(equations, values):
            if x in graph:
                graph[x][y] = v
            else:
                graph[x] = {y: v}
            if y in graph:
                graph[y][x] = 1 / v
            else:
                graph[y] = {x: 1 / v}

        def dfs(s, t):
            """ 计算s到t的路径并计算 叠乘 """
            if s not in graph:
                return -1.
            elif s == t:
                return 1.
            for node in graph[s].keys():
                if node == t:
                    return graph[s][node]
                elif node not in visited:
                    visited.add(node)
                    v = dfs(node, t)        # 直接算s/t不行，那先计算node/t，一步一步最终会计算处s/t
                    if v != -1:
                        return v * graph[s][node]
            return -1

        res = []
        for s, t in queries:
            visited = set()
            v = dfs(s, t)
            res.append(v)
        return res

    def decodeString(self, s: str) -> str:
        """
        394.字符串解码
        2023.07.01 中等
        题解：肯定是要用到栈，重点明白栈内元素的意义
        """
        res, stack, multi = "", [], 0   # res:最终结果 stack:[[multi 数字, res遇到此左括号前的内容], [], ...]
        for c in s:
            if c == '[':    # 遇到'['，记录[multi, res]，并重置这两个变量
                stack.append([multi, res])      # multi重复次数不是指的stack内res，而是当前在拼接的res重复次数
                multi, res = 0, ""
            elif '0' <= c <= '9':       # 字符串s数字可能是多位的，但一个完整的数字的多位肯定是一起出现，这里是处理完整的一个数字；包含'0'，因为有的数字是有0的，和题目并不冲突
                multi = 10 * multi + int(c)
            elif c == ']':  # 遇到']'，开始拼接结果
                cur_multi, last_res = stack.pop()
                res = last_res + res * cur_multi    # 这句可以看出来，stack内的multi指的是当前正在拼接的res重复次数、last_res指的是当前拼接res前面的那部分
            else:
                res += c
        return res

    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        """
        347.前K个高频元素
        2023.07.01 中等
        题解：
        """
        # 所谓桶排序，是什么？https://www.runoob.com/w3cnote/bucket-sort.html
        num2counts = {}
        for n in nums:
            num2counts[n] = num2counts.get(n, 0) + 1
        # counts2num = [[]] * (len(nums) + 1)   # 这是导致结果错误，元素之间会共享内存，修改一个其余都变
        counts2num = [[] for _ in range(len(nums) + 1)]    # 索引代表出现次数，出现次数介于[0,len(nums)]，所以数组长度为len(nums)+1
        for n, counts in num2counts.items():
            counts2num[counts].append(n)
        res = []
        for same_counts_ls in counts2num[::-1]:
            for num in same_counts_ls:
                if len(res) < k:
                    res.append(num)
                else:
                    break
                # 这样代码更简洁
                # res.append(num)
                # if len(res) == k:
                #     return res
        return res

        # 使用内置库
        # from collections import Counter
        #
        # num2counts = sorted(Counter(nums).items(), key=lambda x: x[1], reverse=True)
        # res = []
        # for _ in range(k):
        #     res.append(num2counts.pop(0)[0])
        # return res

    def countBits(self, n: int) -> List[int]:
        """
        338.比特位计数
        2023.07.01 中等
        题解：答案思路妙呀！只需一次简单的遍历
        """
        res = [0] * (n + 1)
        for i in range(1, n + 1):
            if i % 2:
                res[i] = res[i - 1] + 1
            else:
                res[i] = res[i // 2]
        return res

    def rob3(self, root: Optional[TreeNode]) -> int:
        """
        337.打家劫舍III
        2023.07.03 中等
        题解：偷or不偷当前节点
        """
        def robInternal(node):
            if not node:
                return [0, 0]
            leftLs = robInternal(node.left)
            rightLs = robInternal(node.right)
            return [max(leftLs) + max(rightLs),     # 不偷当前节点
                    node.val + leftLs[0] + rightLs[0]]      # 偷当前节点；leftLs[0]指的上一层返回的“不偷当前节点”...
        return max(max(robInternal(root.left)) + max(robInternal(root.right)),          # 不偷root
                   root.val + robInternal(root.left)[0] + robInternal(root.right)[0])   # 偷root

    def coinChange(self, coins: List[int], amount: int) -> int:
        """
        322.零钱兑换
        2023.07.03 中等
        题解：动态规划 dp[n]为最少硬币数  n为目标金额
        """
        # 备忘录memo，消除重复计算
        memo = {}
        def dp(amount):
            if amount in memo:
                return memo[amount]
            if amount == 0:
                return 0
            elif amount < 0:
                return -1
            res = float('inf')
            for coin in coins:
                subproblem = dp(amount - coin)
                if subproblem < 0:
                    continue
                res = min(res, 1 + subproblem)
            memo[amount] = res if res != float('inf') else -1
            return memo[amount]

        return dp(amount)

    def maxProfit(self, prices: List[int]) -> int:
        """
        309.最佳买卖股票时机含冷冻期
        2023.07.03
        题解：dp[i][1、2、3] 每天都有3种状态，不持有：今天啥也没干、持有：今天买入、不持有：今天买入卖出
        """
        # 初始化：只考虑“今天”当天
        # dp[i][0] 不持有，啥都没干的不持有
        # dp[i][1] 持有，今天买入
        # dp[i][2] 不持有，今天买入卖出
        # dp = [[0, -price, 0] for price in prices]
        # # 用昨天的状态 来更新 今天的状态
        # for i in range(1, len(prices)):
        #     # 今天不持有，有2种可能：1.昨天本就不持有 2.昨天卖出
        #     dp[i][0] = max(dp[i - 1][0], dp[i - 1][2])
        #     # 今天持有，有2种可能：1.昨天本就持有 2.昨天没有(非卖出)，今天买入
        #     dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])
        #     # 今天不持有，是因为卖出，所以昨天必须持有今天才有的卖
        #     dp[i][2] = dp[i - 1][1] + prices[i]
        # return max(dp[-1])      # 其实是max(dp[-1][0], dp[-1][2])，仅在 不持有 状态下计算

        # 2023.07.10 复习一下
        dp = [[0, 0, 0] for _ in range(len(prices))]
        dp[0] = [0, -prices[0], 0]
        for i in range(1, len(prices)):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][2])
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])
            dp[i][2] = dp[i - 1][1] + prices[i]
        return max(dp[-1])      # 实际是返回最后一天不持有状态的最大值

        # 超时
        # def dp(amount):
        #     # 递归终止条件
        #     if amount == 0:
        #         return 0
        #     elif amount < 0:
        #         return -1
        #     res = float('inf')
        #     for coin in coins:
        #         subprolem = dp(amount - coin)
        #         if subprolem < 0:   # 无解，继续下一个尝试
        #             continue
        #         res = min(res, 1 + subprolem)
        #     return res if res != float('inf') else -1
        #
        # return dp(amount)

    def lengthOfLIS(self, nums: List[int]) -> int:
        """
        300.最长递增子序列
        2023.07.04 中等
        题解：动态规划 dp[i] 截止到索引i元素的最长递增子序列
        """
        dp = [1] * len(nums)
        for i in range(len(nums)):      # 遍历计算dp[i]
            for j in range(i):      # 截止到索引i嘛
                if nums[j] < nums[i]:   # nums[i]可以接到nums[j]后面
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)

    def findDuplicate(self, nums: List[int]) -> int:
        """
        287.寻找重复数
        2023.07.04 中等
        题解：比较绕，要仔细理解
        """
        left, right = 1, len(nums) - 1      # 在1~n中查找那个重复元素；你都知道mid是可以在范围1~n随机选了，那么left right的初始化也应该符合要求
        while left < right:
            mid = (left + right) // 2       # 随便选个数，恰好数值范围1~n，所以可以利用索引随便选数
            counts = 0      # 计算<=mid的数有多少，(因为数值范围、数值个数的关系，题目已说明只有1个重复数字)
            for n in nums:
                if n <= mid:
                    counts += 1
            if counts > mid:    # counts若大于mid，就是说数值范围1~mid肯定有重复
                right = mid     # 数值mid也可能是重复的怀疑对象
            else:               # 那数值范围1~mid肯定没有重复
                left = mid + 1  # mid也不是怀疑对象
        return right        # 注意这里的返回值；或返回left

    def moveZeroes(self, nums: List[int]) -> None:
        """
        283.移动零
        2023.07.04 简单
        题解：自己想遍历似乎走不通；答案双指针，一次遍历就ok
        """
        ind = 0     # 非零元素该放到的索引位置
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[ind] = nums[i]     # 若nums[ind]是0，覆盖就覆盖了，最后的步骤会把0变成该有的数量
                ind += 1
        for i in range(ind, len(nums)):
            nums[i] = 0

    def numSquares(self, n: int) -> int:
        """
        279.完全平方数
        2023.07.05 中等
        题解：因为使用了j ** 2而不是j * j超时！
        """
        dp = [0] * (n + 1)      # dp[i]表示和为i时所需最少的完全平方数
        for i in range(1, n + 1):   # dp[0]=0嘛
            dp[i] = i       # 先初始化为最大数量，即需要几个1*1相加
            j = 1
            while i - j ** 2 >= 0:
                dp[i] = min(dp[i], 1 + dp[i - j ** 2])
                j += 1
        return dp[n]

    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        """
        240.搜索二维矩阵II
        2023.07.05 中等
        题解：升序、查找--二分查找
        """
        for nums in matrix:
            if target > nums[-1]:
                continue
            elif target < nums[0]:
                return False
            left, right = 0, len(nums) - 1
            while left <= right:
                mid = (left + right) // 2
                if target == nums[mid]:
                    return True
                elif target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
        return False

    def productExceptSelf(self, nums: List[int]) -> List[int]:
        """
        238.除自身以外数组的乘积
        2023.07.05 中等
        题解：先计算当前元素左边所有元素累乘，再计算右边所有元素累乘
        """
        res = [1 for _ in range(len(nums))]
        tmp = 1
        for i in range(len(nums)):
            res[i] = tmp
            tmp *= nums[i]
        tmp = 1
        for i in range(len(nums) - 1, -1, -1):
            res[i] *= tmp
            tmp *= nums[i]
        return res

    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        """
        236.二叉树的最近公共祖先
        2023.07.05 中等
        题解：递归 先序遍历
        """
        if not root or p == root or q == root:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        # 简化一下答案的判断
        if not left: return right
        elif not right: return left
        return root

    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        """
        234.回文链表
        2023.07.05 简单
        题解：step1.找到中间节点
            step2.反转后半部分节点
            step3.计算是否回文
        """
        # 再写一遍
        def end_of_first_half(head):
            # 参数是首节点
            slow = fast = head
            # 如果fast能往后移动2个节点就移动2个节点，否则不移动。
            # 这样无论总节点数为奇数偶数个，slow都会指向前半部分最后一个节点
            while fast.next and fast.next.next:
                slow = slow.next
                fast = fast.next.next
            return slow

        def reverse_list(node):
            # 参数是尚未反转的后半部分第一个节点
            pre_ndoe = None     # 后半部分第一个，反转后成为最后一个，其next是None
            curr_node = node
            while curr_node:
                # 这样理解while，当curr_node是最后一个节点时，过一遍while就明了了，pre_node会指向最后一个节点，curr_node会指向空
                next_node = curr_node.next      # 先记录后一个节点
                curr_node.next = pre_ndoe       # 改变当前节点指向
                pre_ndoe = curr_node
                curr_node = next_node
            return pre_ndoe

        first_half_node = end_of_first_half(head)
        second_half_node = reverse_list(first_half_node.next)   # 后半部分反转后的“首节点”

        result = True
        first_pos = head        # 注意这里初始指向
        second_pos = second_half_node
        # 为什么判断条件包括second_pos，偶数节点两部分长度相同，奇数节点前部分长1个，后半部分先遍历完毕
        while result and second_pos:
            if first_pos.val != second_pos.val:
                result = False
            first_pos = first_pos.next
            second_pos = second_pos.next
        # 后半部分再给人家反转回去，当然不影响通过测试用例
        first_half_node.next = reverse_list(second_half_node)
        return result

        # 答案解法
        # def end_of_first_half(head):
        #     """ 返回前半部分最后一个节点 """
        #     # 无论奇数、偶数个，均能返回前半部分最后一个节点
        #     slow = fast = head
        #     while fast.next and fast.next.next:
        #         slow = slow.next
        #         fast = fast.next.next
        #     return slow
        #
        #     # 偶数个不对
        #     # slow = fast = head
        #     # while fast is not None:
        #     #     slow = slow.next
        #     #     fast = fast.next
        #     #     if fast is not None:
        #     #         fast = fast.next
        #     # return slow
        #
        #
        # def reverse_list(node):
        #     pre_node = None
        #     curr_node = node
        #     while curr_node:
        #         next_node = curr_node.next
        #         curr_node.next = pre_node
        #         pre_node = curr_node
        #         curr_node = next_node
        #     return pre_node
        #
        # first_end_node = end_of_first_half(head)    # 返回前半部分链表的尾结点
        # second_end_node = reverse_list(first_end_node.next)     # 逆转后半部分
        #
        # result = True
        # first_pos = head
        # second_pos = second_end_node
        # while result and second_pos is not None:
        #     if first_pos.val != second_pos.val:
        #         result = False
        #     first_pos = first_pos.next
        #     second_pos = second_pos.next
        # first_end_node.next = reverse_list(second_end_node)
        # return result

        # 空间复杂度O(n)
        # nodes = []
        # while head:
        #     nodes.append(head.val)
        #     head = head.next
        # left, right = 0, len(nodes) - 1
        # while left < right:
        #     if nodes[left] != nodes[right]:
        #         return False
        #     left += 1
        #     right -= 1
        # return True

    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """
        226.翻转二叉树
        2023.07.07 简单
        题解：
        """
        # 层序遍历是可以的
        if not root:
            return root
        one_layer_nodes = [root]
        while one_layer_nodes:
            tmp = []
            while one_layer_nodes:
                node = one_layer_nodes.pop(0)
                node.left, node.right = node.right, node.left
                if node.left:
                    tmp.append(node.left)
                if node.right:
                    tmp.append(node.right)
            one_layer_nodes = tmp
        return root

    def maximalSquare(self, matrix: List[List[str]]) -> int:
        """
        221.最大正方形
        2023.07.07 中等
        题解：dp[i+1][j+1]表示以matrix[i][j]为右下角的最大面积
        """
        m, n = len(matrix), len(matrix[0])
        dp = [[0] * (n + 1) for _ in range(m + 1)]      # 多添加了一行0、一列0，为了方便处理matrix的首行、首列
        maxside = 0
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == '1':
                    # dp的索引0行、0列不用处理，默认是0；主要作用就是为了少判断遇到matrix索引0行、0列的1时
                    dp[i + 1][j + 1] = min(dp[i][j + 1], dp[i + 1][j], dp[i][j]) + 1    # +1的意思，我能利用左、上、左上哪一边的 1
                    maxside = max(maxside, dp[i + 1][j + 1])
        return maxside ** 2

    def findKthLargest(self, nums: List[int], k: int) -> int:
        """
        215.数组中的第k个最大元素
        2023.07.07 中等
        题解：完美通过，当然要细心啦
        """
        # 答案 评论区 提交和测试结果不同，不知道为什么
        # https://leetcode.cn/problems/kth-largest-element-in-an-array/solutions/307351/shu-zu-zhong-de-di-kge-zui-da-yuan-su-by-leetcode-/
        from random import randint

        def quicksort(nums, start, end, target):
            rand = randint(start, end)
            base = nums[rand]
            nums[rand], nums[start] = nums[start], nums[rand]
            index = start
            for i in range(start + 1, end + 1):
                if nums[i] >= base:
                    nums[i], nums[index + 1] = nums[index + 1], nums[i]
                    index += 1      # index最终指向最后一个大于等于base的值
            nums[index], nums[start] = nums[start], nums[index]     # 此时，base左边都是大于等于base的，右边都是小于的
            if index < target:      # base元素的索引index小了，继续递归
                quicksort(nums, index + 1, end, target)
            elif index > target:    # base元素的索引index大了，继续递归
                quicksort(nums, start, index - 1, target)
            # 等于时即找到了，算法停止

        quicksort(nums, 0, len(nums) - 1, k - 1)    # 题目求第k大元素，降序排序后索引为k-1即为所求
        return nums[k - 1]

        # 试一下堆排 但时间复杂度不符合题目要求
        # def HeapSort(nums):
        #     BuildHeap(nums)
        #     res = []
        #     for i in range(len(nums) - 1, 0, -1):
        #         res.append(nums[1])
        #         nums[1], nums[i] = nums[i], nums[1]
        #         AdjustDown(nums, 1, i - 1)
        #     return res
        # def BuildHeap(nums):
        #     k = len(nums) // 2
        #     for i in range(k, 0, -1):
        #         AdjustDown(nums, i, len(nums) - 1)
        #
        # def AdjustDown(nums, k, length):
        #     nums[0] = nums[k]
        #     i = 2 * k
        #     while i <= length:
        #         if i < length and nums[i] > nums[i + 1]:
        #             i = i + 1
        #         if nums[i] < nums[0]:
        #             nums[k] = nums[i]
        #             k = i
        #         i *= 2
        #     nums[k] = nums[0]
        #
        # nums.insert(0, 0)
        # res = HeapSort(nums)
        # print(res)
        # return res[-k]

    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        """
        207.课程表
        2023.07.08 中等
        题解：统计每门课程的入度，字典 前置课程：[课程]
        """
        ind2preNum = [0 for _ in range(numCourses)]     # 初始化每门课程的入度，即所需前置课程的数量
        pre2course = defaultdict(list)      # 前置课程:[课程们]
        for cour, pre in prerequisites:     # 遍历，填充pre2course、ind2preNum
            pre2course[pre].append(cour)
            ind2preNum[cour] += 1
        queue = [i for i, counts in enumerate(ind2preNum) if counts == 0]       # 先学那些不需要前置课程的
        counts = 0      # 已学习的课程数量
        while queue:
            ind = queue.pop()
            counts += 1
            for id in pre2course[ind]:
                if ind2preNum[id] > 0:
                    ind2preNum[id] -= 1
                if ind2preNum[id] == 0:
                    queue.append(id)

            pre2course.pop(ind)
        return counts == numCourses

    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        206.反转链表
        2023.07.08 简单
        题解：还记得 回文链表 吗，用过此题解法
        """
        pre_node = None
        curr_node = head    # 这句似乎多余
        while curr_node:
            next_node = curr_node.next      # 先保留下一个节点
            curr_node.next = pre_node
            pre_node = curr_node
            curr_node = next_node
        return pre_node

    def numIslands(self, grid: List[List[str]]) -> int:
        """
        200.岛屿数量
        2023.07.05 中等
        题解：就记为dfs吧 还算比较好理解
        """
        def dfs(x, y):
            grid[x][y] = 0      # dfs深度优先遍历，只要一调用dfs，就能把所有连着的'1'全都找到并置0，所以题目说主函数中dfs调用次数即岛屿数量
            nr, nc = len(grid), len(grid[0])
            for i, j in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                if 0 <= i < nr and 0 <= j < nc and grid[i][j] == '1':
                    dfs(i, j)

        nr = len(grid)
        if nr == 0:
            return 0
        nc = len(grid[0])
        numIsland = 0
        for i in range(nr):
            for j in range(nc):
                if grid[i][j] == '1':
                    numIsland += 1
                    dfs(i, j)
        return numIsland

    def rob(self, nums: List[int]) -> int:
        """
        198.打家劫舍
        2023.07.08 中等
        题解：能想到动态规划
        """
        dp = [0 for _ in range(len(nums))]
        dp[0] = nums[0]
        for i in range(1, len(nums)):
            dp[i] = max(dp[i - 1], nums[i] + dp[i - 2])
        return max(dp)

    def majorityElement(self, nums: List[int]) -> int:
        """
        169.多数元素
        2023.07.09 简单
        题解：将众数视为正1，负数视为负1，计数器在遍历过程中加减1，计数器等于0时更换候选众数，最终指向就是 众数
            题目要求已说明，出现次数 大于 ⌊ n/2 ⌋，是严格的大于，counts将始终非负
            最后一个方法 https://leetcode.cn/problems/majority-element/solutions/146074/duo-shu-yuan-su-by-leetcode-solution/
        """
        counts = 0
        candidate = 0
        for n in nums:
            if counts == 0:
                candidate = n
            counts += (1 if n == candidate else -1)
        return candidate

    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        """
        160.相交链表
        2023.07.09 简单
        题解：链表A长度a+c, 链表B长度b+c
        """
        PA, PB = headA, headB
        while PA != PB:
            PA = PA.next if PA else headB
            PB = PB.next if PB else headA
        return PA   # or PB

    def maxProduct(self, nums: List[int]) -> int:
        """
        152.乘积最大子数组
        2023.07.09 中等
        题解：能想到动态规划
        """
        # 答案，维护imax imin
        # imax = imin = 1
        # res = float('-inf')     # 记录最大值，初始化为最小
        # for n in nums:
        #     if n < 0:
        #         imax, imin = imin, imax
        #     imax = max(n, imax * n)
        #     imin = min(n, imin * n)
        #     res = max(res, imax)        # 时刻记录imax最大状态，这样就不怕imax在计算过程中被覆盖了
        # return res

        # 答案 使用dp，因为要同时记录遍历过程中最大值和最小值，所以dp是二维
        dp = [[0, 0] for _ in range(len(nums))]     # 同时记录imin、imax
        dp[0] = [nums[0], nums[0]]
        for i in range(1, len(nums)):
            if nums[i] >= 0:
                dp[i][0] = min(nums[i], nums[i] * dp[i - 1][0])     # 求连续累乘子序列明了很多
                dp[i][1] = max(nums[i], nums[i] * dp[i - 1][1])
            else:
                dp[i][0] = min(nums[i], nums[i] * dp[i - 1][1])
                dp[i][1] = max(nums[i], nums[i] * dp[i - 1][0])
        return max([ls[1] for ls in dp])

    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        148.排序链表
        2023.07.09 中等
        题解：归排吧
        """
        def mergesort(head, tail):
            if not head:
                return head
            elif head.next == tail:
                head.next = None
                return head

            slow = fast = head
            while fast != tail:
                slow = slow.next
                fast = fast.next
                if fast != tail:
                    fast = fast.next
            mid = slow
            head1 = mergesort(head, mid)
            head2 = mergesort(mid, tail)
            return merge(head1, head2)

        def merge(head1, head2):
            tmp = resHead = ListNode()
            while head1 and head2:
                if head1.val < head2.val:
                    tmp.next = ListNode(head1.val)
                    tmp = tmp.next
                    head1 = head1.next
                else:
                    tmp.next = ListNode(head2.val)
                    tmp = tmp.next
                    head2 = head2.next

            if head1:
                while head1:
                    tmp.next = ListNode(head1.val)
                    tmp = tmp.next
                    head1 = head1.next
            else:
                while head2:
                    tmp.next = ListNode(head2.val)
                    tmp = tmp.next
                    head2 = head2.next
            return resHead.next

        return mergesort(head, None)

    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        142.环形链表ii
        2023.07.09 中等
        题解：保存节点地址，检查是否有重复
            仔细看下答案 双指针 https://leetcode.cn/problems/linked-list-cycle-ii/solutions/12616/linked-list-cycle-ii-kuai-man-zhi-zhen-shuang-zhi-/
        """
        # 答案 双指针
        # 再写一次
        slow = fast = head
        # 索性，while里面不检查是否能走到空，结束判断是否空
        while fast:     # 链表如果空，循环不执行；fast走得快，所以用fast判断
            slow = slow.next
            fast = fast.next
            if fast:
                fast = fast.next

            if fast == slow:    # slow、fast第一次遇到
                break
        if not fast:    # while是因为fast走到空结束，无环
            return None
        # slow、fast已经第一次遇到，slow还需要a步就能到达入口
        fast = head
        while fast != slow:
            slow = slow.next
            fast = fast.next
        return fast

        # 答案 双指针
        # 自己看了遍代码写的，不太健壮
        # slow = fast = head
        # while True:
        #     if not fast:    # fast能遍历至空，说明无环
        #         return
        #     slow = slow.next
        #     fast = fast.next
        #     if fast:
        #         fast = fast.next
        #
        #     if fast == slow:    # 第一次相遇
        #         break
        # # 这是根据测试用例写出来的，不太健壮
        # if fast is None: return None    # 或 slow is None
        #
        # fast = head
        # while fast != slow:
        #     fast = fast.next
        #     slow = slow.next
        # return fast     # or slow

        # 自己写的
        # if not head:    # 题目说明节点可能是0个
        #     return None
        # address_of_node = set()
        # while head:
        #     if head in address_of_node:
        #         return head
        #     address_of_node.add(head)
        #     head = head.next
        # return None

    def hasCycle(self, head: Optional[ListNode]) -> bool:
        """
        141.环形链表
        2023.07.10
        题解：自己那种保存节点内存地址肯定可以
            试试昨天142.答案的双/快慢指针
        """
        slow = fast = head
        while fast:     # while没有检查二者同时为空的情况
            slow = slow.next
            fast = fast.next
            if fast:
                fast = fast.next

            if fast == slow:    # 第一次相遇 或 都到了结尾
                break

        if not fast:    # slow、fast均为空
            return False
        return True

    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        """
        139.单词拆分
        2023.07.10 中等
        题解：动态规划 dp[i]表示前i个字符能否被wordDict表示
        """
        n = len(s)
        dp = [False for _ in range(n + 1)]
        dp[0] = True
        for i in range(n):      # 用来指示dp，前i个字符是否能被字典表示，若能，考虑i后面若干字符能否被字典表示，一点一点向后蚕食
            for j in range(i + 1, n + 1):
                # dp[i]表示前i个字符，s[i:j]的i表示起始索引，俩i刚好差1 s[i:j]刚好表示前i位之后的若干元素能否被表示
                if dp[i] and s[i:j] in wordDict:     # 前i个字符能否表示已确定，看后面若干字符能否被表示
                    dp[j] = True
        return dp[-1]

    def singleNumber(self, nums: List[int]) -> int:
        """
        136.只出现一次的数字
        2023.07.10 中等
        题解：异或能消掉相同的数
        """
        res = nums.pop(0)
        while nums:
            res ^= nums.pop(0)
        return res

    def longestConsecutive(self, nums: List[int]) -> int:
        """
        128.最长连续序列
        2023.07.10 中等
        题解：尝试从最小的数开始找 哈希加快查找
        """
        nums = set(nums)    # 哈希，加快查找，查一次O(1)
        res = 0
        for n in nums:
            if n - 1 not in nums:   # 保证从最小的数开始找，这样才能找到最长的
                curr_len = 1
                while n + 1 in nums:
                    curr_len += 1
                    n += 1
                res = max(res, curr_len)
        return res

    def maxProfit(self, prices: List[int]) -> int:
        """
        121.买卖股票的最佳时机
        2023.07.10 简单
        题解：一次遍历
        """
        minPrice = float('inf')     # 注意这个初始化，必要的时候初始化为最值
        res = 0
        for n in prices:
            minPrice = min(minPrice, n)
            res = max(res, n - minPrice)
        return res

    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        114.二叉树展开为链表
        2023.07.10 中等
        题解：自己写的，牛逼呀！对每个节点，左子树挪到右子树、右子树放到新子树右子树上
        """
        if not root:
            return
        currNode = root
        while currNode:
            # 没有左节点
            if not currNode.left:
                currNode = currNode.right
                continue
            # 有左节点
            rightNode = currNode.right
            currNode.right = tmp = currNode.left
            currNode.left = None        # 要注意把left置空
            # 把rightNode放到正确的地
            while tmp.right:
                tmp = tmp.right
            tmp.right = rightNode
            currNode = currNode.right

    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        key2ind_inorder = {key: ind for ind, key in enumerate(inorder)}     # 哈希索引，加速查找，O(1)

        def func(preorder_left, preorder_right, inorder_left, inorder_right):
            if preorder_left > preorder_right:      # 相等应该是意味着有1个节点，是可以组成子树的，所以这里停止条件是没有节点
                return None
            rootInd_inorder = key2ind_inorder[preorder[preorder_left]]      # 根节点在中序遍历中的索引
            leftSubTree_len = rootInd_inorder - inorder_left        # 左子树长度
            root = TreeNode(preorder[preorder_left])
            root.left = func(preorder_left + 1, preorder_left + leftSubTree_len,
                             inorder_left, rootInd_inorder - 1)
            root.right = func(preorder_left + leftSubTree_len + 1, preorder_right,
                              rootInd_inorder + 1, inorder_right)
            return root

        n = len(preorder)   # 元素数量，也即节点数量
        return func(0, n - 1, 0, n - 1)



if __name__ == '__main__':
    sl = Solution()

    nums = [4,1,2,1,2]
    print(sl.singleNumber(nums))


    # nums = [3,2,0,-4]
    # head = tmp = ListNode()
    # while nums:
    #     tmp.next = ListNode(nums.pop(0))
    #     tmp = tmp.next
    # head = head.next
    # # 测试打印
    # # while head:
    # #     print(head.val)
    # #     head = head.next
