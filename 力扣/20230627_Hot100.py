from typing import List


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





if __name__ == '__main__':
    sl = Solution()
    s = 'aaa'
    print(sl.countSubstrings(s))