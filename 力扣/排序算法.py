"""
2023.05.07
    对各种排序算法的一次复习 学而时习之
    参考：https://blog.csdn.net/a1159477889/article/details/121168158
"""
import random


# 插入排序
# 7.1.2 直接插入排序
def InsertSort(nums):
    """
    思路：索引0作哨兵，索引1认为有序，从索引2开始遍历
    时间复杂度：O(n^2) 因为2个for循环
    空间复杂度：O(1) 仅哨兵
    稳定性：因为先比较、后移动，比较到小于等于的数时就不移动了，插入--所以是稳定的
    适用性：适用于 顺序存储 和 链式存储结构（大部分排序算法仅适用顺序存储结构）
    :param nums:
    :return:
    """
    if len(nums) < 2:
        return nums
    for i in range(2, len(nums)):
        if nums[i] < nums[i - 1]:
            nums[0] = nums[i]
            for j in range(i - 1, -1, -1):   # 这里范围(i - 1, -1) 的-1很重要，因为要检查到索引0哨兵--哨兵肯定小于等于当前比较的数，所以也可能是插入点，否则可能错过插入点
                if nums[j] > nums[0]:
                    nums[j + 1] = nums[j]
                else:
                    break
            nums[j + 1] = nums[0]
    return nums


# 7.2.2 折半插入排序
def InsertSort02(nums):
    """
    思路：直接插入排序是边比较边移动，在顺序存储下，通常将查找和移动分开来进行优化，减少比较次数(约为o(nlogn))，但没有减少移动次数
    时间复杂度：O(nlogn) 因为折半查找减少了比较次数
    空间复杂度：O(1) 仅哨兵
    稳定性：排序思路还是先查找、后移动，所以依然是稳定的
    适用性：顺序存储结构，因为折半查找进行随机访问
    :param nums:
    :return:
    """
    for i in range(2, len(nums)):   # 同直接插入排序，索引0为哨兵，索引1默认有序，所以从索引2开始遍历
        if nums[i] < nums[i - 1]:
            nums[0] = nums[i]
            low, high = 1, i - 1    # 折半查找的范围
            while low <= high:
                mid = (low + high) // 2
                if nums[mid] < nums[0]:
                    low = mid + 1
                else:
                    high = mid - 1
            for j in range(i - 1, high, -1):    # 需要后移，注意范围
                nums[j + 1] = nums[j]
            nums[high + 1] = nums[0]
    return nums


# 7.2.3 希尔排序
def ShellSort(nums):
    """
    思路：直接插入排序适合数据量少、初始基本有序的场景。为此提出希尔排序，也称缩小增量排序。
        底层思路还是直接插入排序；不使用哨兵了，使用不方便
    时间复杂度：O(n^1.3)~O(n^2)   dk的划分涉及数学未解决的问题；底层逻辑仍然是直接插入排序，因此时间复杂度仍为O(n^2)
    空间复杂度：O(1) 仅tmp临时变量
    稳定性：很明显不稳定，若两个相同的元素划分到不同的子序列中，相对位置可能改变
    适用性：顺序存储，底层仍是直接插入排序 且 有dk划分则需要随机访问，因此仅适用于顺序存储
    :param nums:
    :return:
    """
    dk = len(nums) // 2
    while dk >= 1:
        for i in range(dk, len(nums), dk):      # 从索引dk开始遍历，因为步长是dk了
            if nums[i] < nums[i - dk]:
                tmp = nums[i]   # 不使用哨兵了
                ind = 0     # IDE报错j未定义使用，因此使用此变量记录j
                for j in range(i - dk, -1, dk):
                    if nums[j] > tmp:
                        nums[j + dk] = nums[j]
                        ind = j
                    else:
                        ind = j
                        break
                    if j - dk < 0:
                        ind = j
                        break
                nums[ind + dk] = tmp
        dk -= 1
    return nums


# 交换排序
# 7.3.1冒泡排序
def BubbleSrot(nums):
    """
    思路：从前往后冒大的，则每次冒泡的范围是len(nums) - 1 - i，因为每次冒泡都有一个元素被放到了最终的位置上
    时间复杂度：O(n^2)    若初始有序即最好情况，只需比较n - 1次，此时时间复杂度O(n)；若初始逆序即最坏情况，则此时时间复杂度O(n^2)。通常认为O(n^2)
    稳定性：稳定。只有大于右面的数才往后冒，等于时不发生交换，所以是稳定的
    不同于插入排序，冒泡排序每一次都将一个元素放到了最终地位置上（即冒泡过程产生的子序列是全局有序的）
    :param nums:
    :return:
    """
    for i in range(len(nums) - 1):      # 一共需要冒泡的次数；n-1个元素都放到最终位置上了，则最后那个元素自然也在最终位置上
        flag = False
        for j in range(len(nums) - 1 - i):
            if nums[j] > nums[j + 1]:
                nums[j], nums[j + 1] = nums[j + 1], nums[j]
                flag = True
        if not flag:
            break
    return nums


# 7.3.2快速排序
def QuickSort(nums, low, high):
    """
    【实现错误！已改正】
    思路：先进行划分并返回所选基准的索引，之后分别对索引左右子序列递归快排
        划分算法是将基准放到最终位置上，且小于基准的在左侧、大于基准的在右侧
    时间复杂度：O(n^2)
    空间复杂度：O(logn)~O(n)基准元素选择合适则递归logn次，选择最坏情况则递归n次
    稳定性：不稳定。若两个相同元素被划分到了同一方，则相对位置会发生改变(如两个小于基准的划分到右边)
    不产生有序子序列，因为基准元素选择较随意。但每次排序后基准元素都会放到最终位置上
    :param nums:
    :return:
    """
    def paritition(nums, low, high):
        pivot = nums[low]   # 基准只能选一次
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
        k = paritition(nums, low, high)
        QuickSort(nums, low, k - 1)
        QuickSort(nums, k + 1, high)
    return nums


# 选择排序
# 7.4.1 简单选择排序
def SelectSort(nums):
    """
    思路：当前索引位置i，则从i+1~len(nums)-1依次比较，找出最小的数与索引i交换
    时间复杂度：移动次数倒少了，但比较次数多，仍为O(n^2)
    空间复杂度：O(1) 仅交换时的临时变量
    稳定性：不稳定。选择的时候都会选择最后一个最小的数，若有两个相等且最小的数，肯定是先选中后面的，相对位置就变了
    :param nums:
    :return:
    """
    for i in range(len(nums) - 1):      # 排序次数，因为每一次都会把一个元素放到最终位置上
        minInd = i     # 指示最小数的索引
        for j in range(i, len(nums)):   # 注意寻找范围
            if nums[j] < nums[minInd]:
                minInd = j
        if minInd != i:
            nums[minInd], nums[i] = nums[i], nums[minInd]
    return nums


# # 7.4.2 (大)堆排序
# def HeapSort(nums):
#     """
#     堆排序主方法，下面两个是辅助方法
#     思路：先建(大 or 小)堆，然后每次输出堆顶元素后立刻和最后一个尚未排序元素交换，向下调整。反复，直到排序完毕
#     :param nums:
#     :return:
#     """
#     BuildMaxHeap(nums)
#     for i in range(len(nums) - 1, 0, -1):   # i指示的是：尚未 排序(调节范围内) 的最后一个元素
#         print(nums[1], end=' ')     # 输出堆顶元素
#         nums[1], nums[i] = nums[i], nums[1]     # 交换元素
#         AdjustDown(nums, 1, i - 1)  # 向下调整，因为堆顶换上来了新元素
#
# def BuildMaxHeap(nums):
#     # 建大顶堆，即越靠近堆顶元素越大
#     i = len(nums) // 2
#     while i >= 1:
#         AdjustDown(nums, i, len(nums) - 1)      # 从最后一个非叶结点到第一个非叶结点(注意，索引0是暂存)，依次向下调整，这样一遍就形成了大顶堆
#         i -= 1
#     return nums
#
# def AdjustDown(nums, k, length):
#     nums[0] = nums[k]       # 索引0暂存元素；k为 待调整堆 的根节点
#     i = 2 * k   # i指示k的子节点
#     while i <= length:
#         if i < length and nums[i] < nums[i + 1]:
#             i += 1
#         if nums[i] > nums[0]:
#             nums[k] = nums[i]       # 把i指示的较大元素换到父节点上
#             k = i   # 父节点指向交换的子节点
#         i *= 2      # 这句其实也就是 i = 2 * k
#     nums[k] = nums[0]   # 最后把暂存元素放到索引k上
#     return nums


# 自己写个小顶堆排序
def HeapSort(nums):
    """
    这里实现的是小顶堆，输出从小到大排序
    思路：step1.建堆 step2.输出堆顶元素、换到堆末、向下调整
    :param nums:
    :return:
    """
    BuildMinHeap(nums)
    for i in range(len(nums) - 1, 0, -1):   # i指示 堆 最后一个元素的索引；索引0是暂存，取不到
        print(nums[1], end=' ')
        nums[1], nums[i] = nums[i], nums[1]
        AdjustDown(nums, 1, i - 1)      # 注意，输出时，每次都是输出堆顶元素，因为i指示着堆末元素索引，所以调整范围要减1

def BuildMinHeap(nums):
    """
    思路：从最后一个非叶结点-->第一个非叶结点，依次向下调整
    :param nums:
    :return:
    """
    i = len(nums) // 2
    while i >= 1:
        AdjustDown(nums, i, len(nums) - 1)
        i -= 1

def AdjustDown(nums, k, length):
    """
    向下调整：建堆、排序过程输出 均需要此步骤；向下调整控制着建 大顶堆 or 小顶堆
    :param nums:
    :param k: 待调整的根节点索引
    :param length: 待调整的最后一个节点索引
    """
    nums[0] = nums[k]
    i = 2 * k
    while i <= length:
        if i < length and nums[i] > nums[i + 1]:    # 这里很巧妙，因为要比较i+1，所以让i<length 那么i+1就能<=length了
            i += 1      # i指示k的较大子节点索引
        if nums[i] < nums[0]:
            nums[k] = nums[i]
            k = i
        i *= 2
    nums[k] = nums[0]


# 归并排序
def MergeSort(nums):
    """
    思路：分治法、递归
    时间复杂度：O(nlogn) 合并算法O(n) 递归O(logn)
    空间复杂度：O(n) 不只要递归栈深度，还有合并算法的临时数组，所以是O(n)
    稳定性：稳定
    :param nums:
    :return:
    """
    if len(nums) <= 1:
        return nums
    n = len(nums) // 2
    nums1 = MergeSort(nums[:n])
    nums2 = MergeSort(nums[n:])
    return merge(nums1, nums2)

def merge(nums_1, nums_2):
    i = j = 0
    res = []
    while i < len(nums_1) and j < len(nums_2):
        if nums_1[i] <= nums_2[j]:
            res.append(nums_1[i])
            i += 1
        else:
            res.append(nums_2[j])
            j += 1
    if i >= len(nums_1):
        res += nums_2[j:]
    elif j >= len(nums_2):
        res += nums_1[i:]
    return res


if __name__ == "__main__":
    # random.seed(123)
    nums = [random.randint(1, 100) for _ in range(10)]
    print(f'------------ 插入排序 --------------')
    print(f'{nums[1:]} -- nums')
    print(f'{InsertSort(nums)[1:]} -- InsertSort')
    print(f'{InsertSort02(nums)[1:]} -- InsertSort02')
    print(f'{ShellSort(nums)[1:]} -- ShellSort')
    print(f'------------ 交换排序 --------------')
    nums = [random.randint(1, 100) for _ in range(10)]
    print(f'{nums} -- nums')
    # print(f'{BubbleSrot(nums)} -- BubbleSort')
    print(f'{QuickSort(nums, 0, len(nums) - 1)} -- QuickSort')
    # print(f'{SelectSort(nums)} -- SelectSort')
    # print(f'{HeapSort(nums)} -- HeapSort')
    # print(f'------------ 归并排序 --------------')      # 中间进行排序，会修改nums，所以一个一个排序运行吧
    # print(f'{MergeSort(nums)} -- MergeSort')

