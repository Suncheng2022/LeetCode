from typing import List

class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        """ 77.组合 """
        # 回溯三部曲:
        #   1.递归函数的参数和返回值
        #   2.回溯终止条件
        #   3.单层搜索过程
        # 时间:O(C(n * k)) 空间:O(C(n * k) * k)
        # res = []
        # path = []
        # def backtracking(n, k, startInd):
        #     if len(path) == k:
        #         res.append(path[:])         # res.append(path) 则res中每次保存的是path的引用, 而path在回溯过程中是变化的, 最终res中保存的所有元素都指向同一地址.
        #         return
        #     for i in range(startInd, n + 1):
        #         path.append(i)
        #         backtracking(n, k, i + 1)
        #         path.pop()
        # backtracking(n, k, 1)
        # return res

        # 剪枝
        # path = []
        # res = []
        # def backtracking(n, k, start_ind):
        #     if len(path) == k:
        #         res.append(path[:])
        #         return
        #     for i in range(start_ind, n - (k - len(path)) + 1 + 1):
        #         path.append(i)
        #         backtracking(n, k, i + 1)
        #         path.pop()
        # backtracking(n,k,1)
        # return res

        # Again
        # 要点: 一个集合求组合使用start_ind, 目的是每层不重复使用元素
        path = []
        res = []
        def backtracking(n, k, start_ind):
            if len(path) == k:
                res.append(path[:])
                return
            for i in range(start_ind, n - (k - len(path)) + 1 + 1):     # 剪枝
                path.append(i)
                backtracking(n, k, i + 1)
                path.pop()
        backtracking(n, k, 1)
        return res

    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        """ 216.组合总和III """
        # path = []
        # res = []
        # def backtracking(k, n, start_ind):
        #     if sum(path) == n and len(path) == k:
        #         res.append(path[:])
        #         return
        #     for i in range(start_ind, 9 + 1):
        #         path.append(i)
        #         backtracking(k, n, i + 1)
        #         path.pop()
        # backtracking(k, n, 1)
        # return res

        # 剪枝
        # 其实只需改动for
        # path = []
        # res = []
        # def backtracking(k, n, start_ind):
        #     if len(path) == k:
        #         if sum(path) == n:
        #             res.append(path[:])
        #         return
        #     for i in range(start_ind, 9 - (k - len(path)) + 1 + 1):     # 两个+1: 第一个+1表示最晚的起始索引, 第二个+1是python语法需要
        #         path.append(i)
        #         backtracking(k, n, i + 1)
        #         path.pop()
        # backtracking(k, n, 1)
        # return res

        # Again
        # 要点: 一个集合求组合要使用start_ind, 目的是每层不使用重复元素
        path = []
        res = []
        def backtracking(k, n, start_ind):
            if sum(path) > n:       # 剪枝
                return
            if len(path) == k:
                if sum(path) == n:
                    res.append(path[:])
                return
            for i in range(start_ind, 9 + 1):
                path.append(i)
                backtracking(k, n, i + 1)
                path.pop()
        backtracking(k, n, 1)
        return res
    
    def letterCombinations(self, digits: str) -> List[str]:
        """ 17.电话号码的字母组合 \n
            77.组合 和 216.组合总和III 是求同一个集合的组合, 而本题是求不同集合的组合--区别是start_ind的作用 """
        # digit2letter = {
        #     2: "abc", 
        #     3: "def",
        #     4: "ghi",
        #     5: "jkl",
        #     6: "mno",
        #     7: "pqrs",
        #     8: "tuv",
        #     9: "wxyz"
        # }
        # s = ""
        # res = []
        # def backtracking(digits, ind):
        #     nonlocal s
        #     if ind == len(digits):
        #         res.append(s)
        #         return
        #     digit = int(digits[ind])
        #     letters = digit2letter[digit]
        #     for i in range(len(letters)):                
        #         s += letters[i]
        #         backtracking(digits, ind + 1)
        #         s = s[:-1]
        # if len(digits) == 0:
        #     return res
        # backtracking(digits, 0)
        # return res

        # Again
        # 要点: 多个集合求组合, 不使用start_ind. 题目中使用的ind是指向不同集合
        dict2letter = {
            2: 'abc', 3: 'def',
            4: 'ghi', 5: 'jkl', 6: 'mno',
            7: 'pqrs', 8: 'tuv', 9: 'wxyz'
        }
        path = ''
        res = []
        def backtracking(digits, ind):
            nonlocal path
            if ind == len(digits):
                res.append(path)
                return
            letters = dict2letter[int(digits[ind])]
            for i in range(len(letters)):
                path += letters[i]
                backtracking(digits, ind + 1)
                path = path[:-1]
        if len(digits) == 0:
            return res
        backtracking(digits, 0)
        return res


    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        """ 39.组合总和 \n
            什么时候需要start_ind: \n
                1.对一个集合求组合需要start_ind
                2.对多个集合求组合不需要start_ind
                3.元素可重复使用, 递归时start_ind不加一 """
        # path = []
        # res = []
        # def backtracking(candidates, target, start_ind):
        #     if sum(path) > target:
        #         return
        #     elif sum(path) == target:
        #         res.append(path[:])
        #         return
        #     for i in range(start_ind, len(candidates)):
        #         path.append(candidates[i])
        #         backtracking(candidates, target, i)         # 可重复选取, start_ind不加一
        #         path.pop()
        # backtracking(candidates, target, 0)
        # return res

        # 剪枝
        # path = []
        # res = []
        # def backtracking(candidates, target, start_ind):
        #     if sum(path) > target:
        #         return
        #     elif sum(path) == target:
        #         res.append(path[:])
        #         return
        #     candidates.sort()
        #     for i in range(start_ind, len(candidates)):
        #         if sum(path) + candidates[i] > target:      # 在这里进行剪枝, 前提是candidates有序. 可以思考一下树形结构, 代码在这里是对for即横向的剪枝
        #             return
        #         path.append(candidates[i])
        #         backtracking(candidates, target, i)
        #         path.pop()
        # backtracking(candidates, target, 0)
        # return res

        # Again
        # 要点: !!! 元素可重复使用 -- start_ind不加1, 表示下一层还可选取当前元素, 但还是要使用start_ind, 因为要保证没有重复组合, 一个集合求组合就是要用start_ind !!!
        path = []
        res = []
        def backtracking(start_ind):
            if sum(path) > target:
                return
            if sum(path) == target:
                res.append(path[:])
                return
            for i in range(start_ind, len(candidates)):
                path.append(candidates[i])
                backtracking(i)
                path.pop()
        backtracking(0)
        return res
    
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        """ 40.组合总和II \n
            注意:
                candidates元素有重复, 重复元素的处理逻辑要注意, 如何完成去重的
                是不同树层的去重[要求有序], 还是不同树枝的去重 """
        # path = []
        # res = []
        # used = [False] * len(candidates)
        # def backtracking(candidates, target, start_ind):
        #     if sum(path) == target:
        #         res.append(path[:])
        #         return
        #     for i in range(start_ind, len(candidates)):
        #         if i > 0 and candidates[i - 1] == candidates[i] and used[i - 1] == False:       # 去重
        #             continue
        #         if sum(path) + candidates[i] > target:
        #             return
        #         used[i] = True
        #         path.append(candidates[i])
        #         backtracking(candidates, target, i + 1)
        #         path.pop()
        #         used[i] = False
        # candidates.sort()
        # backtracking(candidates, target, 0)
        # return res

        # Again
        # 要点: 1.元素有重复, 要去重 
        #      2.一个集合求组合, 要使用start_ind
        path = []
        res = []
        used = [False] * len(candidates)
        def backtracking(start_ind):
            # if sum(path) > target:                        # 剪枝一, 在递归内剪枝
            #     return
            if sum(path) == target:
                res.append(path[:])
                return
            for i in range(start_ind, len(candidates)):
                if i > 0 and candidates[i - 1] == candidates[i] and used[i - 1] == False:
                    continue
                if sum(path) + candidates[i] > target:      # 剪枝二, 在递归外剪枝
                    return
                path.append(candidates[i])
                used[i] = True
                backtracking(i + 1)
                used[i] = False
                path.pop()
        candidates.sort()
        backtracking(0)
        return res
    
    def partition(self, s: str) -> List[List[str]]:
        """ 131.分割回文串 """
        # def isParamlim(s):
        #     left, right = 0, len(s) - 1
        #     while left <= right:
        #         if s[left] != s[right]:
        #             return False
        #         left += 1
        #         right -= 1
        #     return True
        
        # path = []
        # res = []
        # def backtracking(s, start_ind):
        #     if start_ind >= len(s):
        #         res.append(path[:])
        #         return
        #     for i in range(start_ind, len(s)):
        #         if isParamlim(s[start_ind:i+1]):
        #             path.append(s[start_ind:i+1])
        #         else:
        #             continue
        #         backtracking(s, i + 1)
        #         path.pop()
        # backtracking(s, 0)
        # return res

        # Again
        # 分割问题 也是 组合问题
        path = []
        res = []
        def isValid(s):
            left, right = 0, len(s) - 1
            while left <= right:
                if s[left] != s[right]:
                    return False
                left += 1
                right -= 1
            return True

        def backtracking(start_ind):
            if start_ind == len(s):
                res.append(path[:])
                return
            for i in range(start_ind, len(s)):
                sub = s[start_ind:i + 1]
                if isValid(sub):
                    path.append(sub)
                    backtracking(i + 1)
                    path.pop()
        backtracking(0)
        return res
    
    def restoreIpAddresses(self, s: str) -> List[str]:
        """ 93.复原IP地址 \n
            类似 131.分割回文串, 分割问题 类似 组合问题 """
        # Again
        # 要点: 分割问题 也是 组合问题
        path = []
        res = []
        def isValid(s):
            if len(s) == 1 and (s[0] < '0' or s[0] > '9'):
                return False
            elif len(s) > 1 and (s[0] <= '0' or s[0] > '9'):
                return False
            num = 0
            for c in s:
                if c < '0' or c > '9':
                    return False
                num = num * 10 + int(c)
            return num <= 255
        
        def backtracking(start_ind):
            if len(path) > 4:
                return
            elif len(path) == 4:
                if start_ind == len(s):
                    res.append('.'.join(path[:]))
                return
            for i in range(start_ind, len(s)):
                sub = s[start_ind:i + 1]
                if isValid(sub):
                    path.append(sub)
                    backtracking(i + 1)
                    path.pop()
        backtracking(0)
        return res

        # Again_
        # path = []
        # res = []
        # def backtracking(s, start_ind):
        #     if len(path) == 4:
        #         if start_ind == len(s):
        #             res.append('.'.join(path))
        #         return
        #     for i in range(start_ind, len(s)):
        #         tmp = s[start_ind:i + 1]
        #         # 判断合法
        #         if len(tmp) == 1 and (tmp[0] < '0' or tmp[0] > '9'):
        #             return
        #         elif len(tmp) > 1 and tmp[0] == '0':
        #             return
        #         elif int(tmp) < 0 or int(tmp) > 255:
        #             return
        #         path.append(tmp)
        #         backtracking(s, i + 1)
        #         path.pop()
        # backtracking(s, 0)
        # return res

        # 不容易呀, 终止条件和判断合法花费时间较多
        # path = []
        # res = []
        # def backtracking(s, start_ind):
        #     if len(path) == 4:
        #         if start_ind == len(s):
        #             res.append(".".join(path))
        #         return
        #     for i in range(start_ind, len(s)):
        #         sub = s[start_ind:i + 1]
        #         if len(sub) == 1 and (sub[0] < '0' or sub[0] > '9'):
        #             break
        #         elif len(sub) > 1 and sub[0] == '0':
        #             break
        #         elif int(sub) > 255:
        #             break
        #         path.append(sub)
        #         backtracking(s, i + 1)
        #         path.pop()
        # backtracking(s, 0)
        # return res
    
    def subsets(self, nums: List[int]) -> List[List[int]]:
        """ 78.子集 \n
            收集所有节点 """
        # path = []
        # res = []
        # def backtracking(nums, start_ind):
        #     res.append(path[:])
        #     # 终止条件
        #     if start_ind >= len(nums):
        #         return
        #     # 单层递归
        #     for i in range(start_ind, len(nums)):
        #         path.append(nums[i])
        #         backtracking(nums, i + 1)
        #         path.pop()
        # backtracking(nums, 0)
        # return res

        # Again
        # 要点: 子集收集所有节点, 组合收集叶节点
        path = []
        res = []
        def backtracking(start_ind):
            res.append(path[:])
            for i in range(start_ind, len(nums)):
                path.append(nums[i])
                backtracking(i + 1)
                path.pop()
        backtracking(0)
        return res
    
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        """ 90.子集II \n
            推测去重逻辑使用used, 对, 即 40.组合总和II 的去重 """
        # path = []
        # res = []
        # used = [False] * len(nums)
        # def backtracking(nums, start_ind):
        #     res.append(path[:])
        #     for i in range(start_ind, len(nums)):
        #         if i > 0 and nums[i - 1] == nums[i] and used[i - 1] == False:
        #             continue
        #         path.append(nums[i])
        #         used[i] = True
        #         backtracking(nums, i + 1)
        #         used[i] = False
        #         path.pop()
        # nums.sort()
        # backtracking(nums, 0)
        # return res

        # Again
        # 要点: 去重而已
        path = []
        res = []
        used = [False] * len(nums)
        def backtracking(start_ind):
            res.append(path[:])
            for i in range(start_ind, len(nums)):
                if i > 0 and nums[i - 1] == nums[i] and used[i - 1] == False:
                    continue
                path.append(nums[i])
                used[i] = True
                backtracking(i + 1)
                used[i] = False
                path.pop()
        nums.sort()
        backtracking(0)
        return res

    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        """ 491.非递减子序列 \n
            看起来与 90.子集II 相似, 其实不同 \n
            不能排序, 否则都为非递减序列了 """
        # path = []
        # res = []
        # def backtracking(nums, start_ind):
        #     if len(path) > 1:
        #         res.append(path[:])
        #         # 不加终止条件, 因为要遍历所有可能节点
        #     unuse = set()                               # 用于去重(不能使用used数组了, 因为不能排序嘛)
        #     for i in range(start_ind, len(nums)):
        #         if (len(path) and path[-1] > nums[i]) or (nums[i] in unuse):    # 同层去重
        #             continue
        #         unuse.add(nums[i])
        #         path.append(nums[i])
        #         backtracking(nums, i + 1)
        #         path.pop()
        # backtracking(nums, 0)
        # return res

        # Again
        # 要点: 不能排序, 通过集合完成层上的去重
        path = []
        res = []
        def backtracking(start_ind):
            if len(path) >= 2:
                res.append(path[:])
            used = set()
            for i in range(start_ind, len(nums)):
                if (nums[i] in used) or (len(path) and path[-1] > nums[i]):
                    continue
                path.append(nums[i])
                used.add(nums[i])           # 没有删除回溯, 好奇么? 只负责本层, 所以不需要回溯
                backtracking(i + 1)
                path.pop()
        backtracking(0)
        return res
    
    def permute(self, nums: List[int]) -> List[List[int]]:
        """ 46.全排列 """
        # 不使用start_ind, 因为每层都要从头开始搜索
        # 使用used数组记录path放了哪些元素
        # path = []
        # res = []
        # used = [False] * len(nums)
        # def backtracking(nums, used):
        #     if len(path) == len(nums):
        #         res.append(path[:])
        #         return
        #     for i in range(0, len(nums)):
        #         if used[i]:
        #             continue
        #         used[i] = True
        #         path.append(nums[i])
        #         backtracking(nums, used)
        #         path.pop()
        #         used[i] = False
        # backtracking(nums, used)
        # return res

        path = []
        res = []
        used = []      # 树枝去重. 题目要求说明无重复, 使用bool也可
        def backtracking():
            if len(path) == len(nums):
                res.append(path[:])
                return
            for i in range(len(nums)):
                if nums[i] in used:
                    continue            # 写return的话, 一层的所有分支都被砍掉, 会漏解
                used.append(nums[i])
                path.append(nums[i])
                backtracking()
                path.pop()
                used.pop()

        backtracking()
        return res
    
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        """ 47.全排列II \n
            排序 去重 """
        # path = []
        # res = []
        # used = [False] * len(nums)
        # def backtracking(nums):
        #     if len(path) == len(nums):
        #         res.append(path[:])
        #         return
        #     for i in range(len(nums)):
        #         if i and nums[i - 1] == nums[i] and used[i - 1] == False:
        #             continue
        #         if used[i] == False:        # 保证不重复使用元素. 之前组合问题通过start_ind完成了类似功能
        #             used[i] = True
        #             path.append(nums[i])
        #             backtracking(nums)
        #             path.pop()
        #             used[i] = False
        
        # nums.sort()
        # backtracking(nums)
        # return res

        # Again
        # 要点: nums有重复
        path = []
        res = []
        used = [False] * len(nums)
        nums.sort()
        def backtracking():
            if len(path) == len(nums):
                res.append(path[:])
                return
            for i in range(len(nums)):
                if i > 0 and nums[i - 1] == nums[i] and used[i - 1] == False:
                    continue
                path.append(nums[i])
                used[i] = True
                backtracking()
                used[i] = False
                path.pop()
        backtracking()
        return res
    
if __name__ == "__main__":
    sl = Solution()

    nums = [1, 1, 2]
    print(sl.permuteUnique(nums))
"""
回溯三部曲:
    1.递归函数的参数和返回值
    2.回溯终止条件
    3.单层搜索过程
"""
# 性能分析:https://programmercarl.com/%E5%91%A8%E6%80%BB%E7%BB%93/20201112%E5%9B%9E%E6%BA%AF%E5%91%A8%E6%9C%AB%E6%80%BB%E7%BB%93.html#%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90
# 子集问题: 
#   时间: 每个元素无外乎选与不选, 共2^n个状态. 将每一次结果放进res时间复杂度为O(n) --> 所以, 时间复杂度不会超过O(n * n^2)
#   空间: 递归深度为n, 每一层递归所用空间为常量 --> 所以O(n)
# 组合问题: 其实类似子集问题
# 排列问题:
#   时间: 第一层节点为n, 下一层每个节点延续n-1个节点, 下一层每个节点延续n-2个节点. 将每一次结果放进res时间复杂度为O(n) --> 所以 O(n! * n), 简化为O(n!)
#   空间: 同子集问题 O(n)