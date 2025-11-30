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

if __name__ == '__main__':
    sl = Solution()

    nums = [3,3]
    target = 6
    print(sl.twoSum(nums, target))