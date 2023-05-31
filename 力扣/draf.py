"""
2023.05.27 复习排序算法
"""
import random
import copy

def InsertSort(nums):
    """
    直接插入排序，取消了“哨兵”
    时间复杂度：O(n^2)
    空间复杂度：O(1) 临时变量
    稳定性：稳定，先比较后移动，所以是稳定的
    """
    for i in range(1, len(nums)):
        if nums[i - 1] > nums[i]:
            tmp = nums[i]
            for j in range(i - 1, -1, -1):
                if nums[j] > tmp:
                    nums[j + 1] = nums[j]
                else:
                    j += 1      # 取消了“哨兵”， 所以定位到插入位置后要+1，方便后面直接赋值
                    break
            nums[j] = tmp
    return nums


def InsertSort02(nums):
    """
    折半插入排序，寻找插入位置时使用二分查找减少比较次数，但不减少移动次数；取消“哨兵”；注：二分查找nums[mid]==nums[i]后并没有break
    时间复杂度：O(nlogn) 折半查找减少了比较次数
    空间复杂度：O(1)
    稳定性：稳定
    """
    for i in range(1, len(nums)):
        if nums[i - 1] > nums[i]:
            tmp = nums[i]
            l, r = 0, i - 1
            while l <= r:
                m = (l + r) // 2
                if nums[m] < tmp:
                    l = m + 1
                else:
                    r = m - 1
            # 索引l也向后移动，后面可以直接对索引l赋值了
            for j in range(i - 1, l - 1, -1):
                nums[j + 1] = nums[j]
            nums[l] = tmp
    return nums


def ShellSort(nums):
    """
    希尔排序  直接插入排序适合数据量少、初始基本有序，为此提出希尔排序，也叫缩小增量排序
    时间复杂度：O(n^1.3)~O(n^2) dk的划分是数学上未解决的难题，暂时认为是O(n^2)吧
    空间复杂度：O(1)
    稳定性：不稳定，两个相同的数划分到了不同的子序列中，相对顺序可能就改变了
    """
    dk = len(nums) // 2
    while dk:
        for i in range(dk, len(nums), dk):
            if nums[i - dk] > nums[i]:
                tmp = nums[i]
                for j in range(i - dk, -1, -dk):
                    if nums[j] > tmp:
                        nums[j + dk] = nums[j]
                    else:
                        j += dk
                        break
                nums[j] = tmp
        dk -= 1
    return nums


def BubbleSort(nums):
    """
    冒泡排序  尤其注意第二个for的取值范围！
    时间复杂度：O(n^2)
    空间复杂度：O(1)  交换两个变量时
    稳定性：稳定，只有相邻的两个数是小于关系才交换，其余情况不交换
    全局有序
    """
    for i in range(len(nums) - 1):      # 冒泡次数
        flag = False
        for j in range(1, len(nums) - i):   # 这个范围一定要注意；索引j 与 索引j+1 比较
            if nums[j - 1] > nums[j]:
                nums[j - 1], nums[j] = nums[j], nums[j - 1]
                flag = True
        if not flag:
            break
    return nums


def QuickSort(nums, low, high):
    """
    快速排序 发现了之前实现是错误的，这里是正确的
    时间复杂度：O(n^2)  容易记错哦
    空间复杂度：O(logn)~O(n)   递归调用；基准选择合适递归logn次，选择不合适则递归n次
    稳定性：不稳定，如果两个较小的数位于基准的右边，调整后相对位置改变
    全局有序
    """
    def Paritation(nums, low, high):
        pivot = nums[low]
        while low < high:
            while low < high and nums[high] >= pivot:
                high -= 1
            nums[low] = nums[high]
            while low < high and nums[low] <= pivot:
                low += 1
            nums[high] = nums[low]
        nums[low] = pivot
        return low

    if low < high:
        k = Paritation(nums, low, high)
        QuickSort(nums, low, k - 1)
        QuickSort(nums, k + 1, high)
    return nums


def SelectSort(nums):
    """
    简单选择排序
    时间复杂度：O(n^2)
    空间复杂度：O(1)  使用常数个空间
    稳定性：不稳定，每一次选择的时候会选择最后一个较小的数，这样相对位置就变了
    其实这里可以优化掉minVal，记录了minInd就足够了，对吧
    """
    for i in range(len(nums) - 1):      # 进行简单选择的次数，选择len(nums) - 1次即可
        minVal, minInd = float('inf'), -1
        for j in range(i, len(nums)):   # 每次进行简单选择的范围，要包括i，因为选择的结果要放到索引i，索引i也要参加比较
            if nums[j] < minVal:
                minVal = nums[j]
                minInd = j
        nums[i], nums[minInd] = nums[minInd], nums[i]
    return nums



if __name__ == "__main__":
    nums = [random.randint(0, 20) for _ in range(10)]
    print(nums)
    print(InsertSort(copy.deepcopy(nums)), 'InsertSort')
    print(InsertSort02(copy.deepcopy(nums)), 'InsertSort02')
    print(ShellSort(copy.deepcopy(nums)), 'ShellSort')
    print(BubbleSort(copy.deepcopy(nums)), 'BubbleSort')
    print(QuickSort(copy.deepcopy(nums), 0, len(nums) - 1), 'QuickSort')
    print(SelectSort(copy.deepcopy(nums)), 'SelectSort')