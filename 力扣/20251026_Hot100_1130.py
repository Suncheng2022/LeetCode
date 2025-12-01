from typing import List, Optional

# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

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
        res = [0] * len(temperatures)
        stack = []
        for i, t in enumerate(temperatures):
            while stack and t > temperatures[stack[-1]]:
                ind = stack.pop()
                res[ind] = i - ind
            stack.append(i)
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


if __name__ == '__main__':
    sl = Solution()

    nums = [3,2,1,0,4]
    print(sl.canJump(nums))