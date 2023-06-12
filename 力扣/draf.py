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


def HeapSort(nums):
    """
    堆排序 索引0作哨兵
    时间复杂度：O(nlogn)  建堆需要O(n)
    空间复杂度：O(1)
    稳定性：不稳定。建堆的过程就可能不稳定，在比较nums[i]与nums[i+1]时
    """
    BuildHeap(nums)
    for i in range(len(nums) - 1, 0, -1):   # i指示有效堆的最后一个元素
        print(nums[1], end=', ')
        nums[1], nums[i] = nums[i], nums[1]
        AdjustDown(nums, 1, i - 1)
    # return nums   # 最终的返回结果是依次取nums[1]，似乎不能直接返回nums; 可以直接返回！！！

def BuildHeap(nums):
    i = len(nums) // 2
    while i >= 1:   # 索引0是哨兵
        AdjustDown(nums, i, len(nums) - 1)
        i -= 1

def AdjustDown(nums, k, length):
    """ 向下调整/自上而下调整, 控制从小到大排序还是从大到小排序
        :param k: 待向下调整的节点索引
        :param length: 有效堆的长度 """
    nums[0] = nums[k]
    i = 2 * k
    while i <= length:
        if i < length and nums[i] > nums[i + 1]:
            i += 1
        if nums[i] < nums[0]:   # 为什么？？？——自己在脑子想象一下向下调整的动态图，是把索引k节点往下不断地比较移动
            nums[k] = nums[i]
            k = i
        i *= 2
    nums[k] = nums[0]


def MergeSort(nums):
    """
    归并排序 我这应该是自底向上的归排吧，我觉得是...
    时间复杂度：O(nlogn)
    空间复杂度：O(n)  递归占用的栈、进行merge时使用的临时数组
    稳定性：不稳定，设想一个示例，两个相等且挨着的数被分到挨着的两组中，极有可能排序后调换了位置
    """
    if len(nums) == 1:
        return nums
    mid = len(nums) // 2
    nums1 = MergeSort(nums[:mid])
    nums2 = MergeSort(nums[mid:])
    return merge(nums1, nums2)

def merge(nums1, nums2):
    i, j = 0, 0
    res = []
    while i < len(nums1) and j < len(nums2):
        if nums1[i] <= nums2[j]:
            res.append(nums1[i])
            i += 1
        else:
            res.append(nums2[j])
            j += 1
    if i < len(nums1):
        res.extend(nums1[i:])
    if j < len(nums2):
        res.extend(nums2[j:])
    return res


if __name__ == "__main__":
    # random.seed(1)    # 随机种子，固定每次运行产生的随机数
    nums = [random.randint(0, 20) for _ in range(10)]
    print(nums)
    print(InsertSort(copy.deepcopy(nums)), 'InsertSort')
    print(InsertSort02(copy.deepcopy(nums)), 'InsertSort02')
    print(ShellSort(copy.deepcopy(nums)), 'ShellSort')
    print(BubbleSort(copy.deepcopy(nums)), 'BubbleSort')
    print(QuickSort(copy.deepcopy(nums), 0, len(nums) - 1), 'QuickSort')
    print(SelectSort(copy.deepcopy(nums)), 'SelectSort')
    print(HeapSort(copy.deepcopy(nums)), 'HeapSort')
    print(MergeSort(copy.deepcopy(nums)), 'MergeSort')