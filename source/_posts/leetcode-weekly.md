---
title: Leetcode Weekly
date: 2020-06-14 17:19:18
tags: Coding
Categories: Coding
---

Leetcode每周碎碎念——参加了一个训练计划，每周输出一个Leetcode题目，除了有特别想总结的专题，通常会把内容都倒到这里。



<!--more-->

## [Gas Station](https://leetcode.com/problems/gas-station/)

### 题目描述

你要旅行经过N个加油站，这N个加油站组成一个圈（环路）。给两个数组，一个是`gas`代表每个加油站各有多少单位的油，另一个是`cost`代表从某个加油站到它下一个加油站需要消费多少油。

要求如果你能从某个加油站出发绕N个加油站一圈的话，返回这个加油站的index，否则返回-1

例子：

> 输入：
>
> gas  = [1,2,3,4,5]
> cost = [3,4,5,1,2]
>
> 输出：3
>
> 解释:
> Start at station 3 (index 3) and fill up with 4 unit of gas. Your tank = 0 + 4 = 4
> Travel to station 4. Your tank = 4 - 1 + 5 = 8
> Travel to station 0. Your tank = 8 - 2 + 1 = 7
> Travel to station 1. Your tank = 7 - 3 + 2 = 6
> Travel to station 2. Your tank = 6 - 4 + 3 = 5
> Travel to station 3. The cost is 5. Your gas is just enough to travel back to station 3.
> Therefore, return 3 as the starting index.

### 解法

能肯定的一点是，如果所有的gas加起来没有cost多的话，那么肯定是不能绕一周的。那么反过来，如果gas的和不小于cost，是否能保证就有解呢？用题目中的例子画个图，发现确实是这样。

首先合并下两个数组，从 `gas` 减去 `cost`，那么新的数就代表从加油站 i 想去到 i+1，到达 i 的时候车上至少要有多少油，也可以理解为需要消耗多少油 （负为消耗，正为可以剩余）。

按照上边的说法，如果从第0个加油站出发，可以画出如下的图，横坐标代表加油站的index，纵坐标代表从第零个加油站到当前加油站 i ，一开始 需要多少油才能要到达下一个 i+1。比如在第0个的时候，需要 $gas[0] - cost[0] = 1 - 3 = -2$，在第1个的时候也需要 -2，但因为是从零出发的，原本已经需要-2了，所以在1的时候就累计需要-4了。

![20200621173924573](/img/post_img/20200621173924573.png)

可以看到，在第二个加油站的时候需要的油最多，达到了-6，之后需要的油就不断减少了--即折线上升。因为总的油是够的，那如果从最低点开始（即第二个加油站），不就能把整个折线图往上推到直线 y = 0 上吗？

确实如此，因为你是从最低点开始的，那个时候是0，那么后面怎么折腾也不会再跌落到0。

假设从第二个加油站出发，可以得到一个新的图如下：

![20200621175530790](/img/post_img/20200621175530790.png)

所以，这题其实只要先判断下总的油够不够，够的话就找到油最少的那个index i，返回 i+1就行了，可行的代码如下

```python
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        if sum(gas) - sum(cost) < 0: return -1

        stock = idx = 0
        minimun = float('inf')
        for i, (g, c) in enumerate(zip(gas, cost)):
            stock += g - c
            if stock < minimun:
                minimun = stock
                idx = i + 1
        return idx % len(gas)
```

另外一种写法是，每次遇到存油小于0的（即从当前规定的起点不可行），就把开始的index设置为下一个，然后stock置为零，因为无论如何也有一个index是可行的，所以在这个index前面的不需要考虑。

```python
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        if sum(gas) - sum(cost) < 0: return -1

        stock = idx = 0
        for i, (g, c) in enumerate(zip(gas, cost)):
            stock += g - c
            if stock < 0:
                stock = 0
                idx = i + 1
        return idx
```





## Word Break I/II

 ### 题目描述

给一个字典**wordDict**里面有一些单词，然后再给一个字符串**s**，想让你

- Word Break I: 判断 **s** 是不是能够由 **wordDict**里面的词构成
  - Input: s = "leetcode", wordDict = ["leet", "code"]
  - Output: True
    - because "leetcode" can be separated into "leet" + "code"
- Word Break II: 找出所有可能的构成方式
  - Input: s = "catsanddog" wordDict = ["cat", "cats", "and", "sand", "dog"]
  - Output: ["cats and dog", "cat sand dog"]
    - 有两种构成方式，用空格隔开

### 解法

这题是个经典的记忆化搜索题目，bottom up可以每次搜`s[i:]`，从后往前搜，用dp记录中间结果降低复杂度; Top down用递归同理。下面贴 Word Break II的代码，I同理

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        wordDict = set(wordDict)
        @lru_cache(None)
        def dfs(begin):
            if begin >= len(s): return [""]
            res = []
            for end in range(begin+1, len(s)+1):
                if s[begin:end] in wordDict:
                    head = (" " if begin else "") + s[begin:end]
                    res.extend([head + item for item in dfs(end)])
            return res
        return dfs(0)
```

> - 在写top down时，这里python有个很好用的修饰器是 `lru_cache`
>
> - 它帮你记住当前参数的返回结果，下次函数被相同的参数调用时，会直接返回结果
> - 其实也就是帮你自动化一个dp的过程，具体其它用法可以参考[python文档](https://docs.python.org/3/library/functools.html#functools.lru_cache)

## Candy Crush

### 题目描述

这题描述很简单，就是模拟完成糖果消消乐的消除过程 -- 

- 用一个面板来表示糖果网格的2D整数数组，不同的正整数board[i][j]表示不同类型的糖果，值board[i][j] = 0表示位置(i,j)处的单元格为空
- 如果三个或三个以上相同类型的糖果在垂直或水平方向相邻，则将它们同时“压碎”-这些位置将变为空。
- 在将所有糖果同时压碎后，如果板上的空白区域上方有糖果，则这些糖果将掉落，直到它们同时撞到糖果或底部。（没有新的糖果会掉到顶部边界之外。）
- 完成上述步骤后，可能会存在更多可以被压碎的糖果。如果是这样，则需要重复以上步骤。
- 如果不存在更多可以压碎的糖果（例如，板子稳定），则返回当前的板子。

例：

> 输入：
> board = [[110,5,112,113,114]，[210,211,5,213,214]，[310,311,3,313,314]，[410,411,412,5,414]，[5,1,512,3,3]，[610,4,1,613,614]，[710,1,2,713,714] ，[810,1,2,1,1]，[1,1,2,2,2]，[4,1,4,4,1014]]


> 输出：
> [[0,0,0,0,0]，[0,0,0,0,0]，[0,0,0,0,0]，[110,0,0,0,114]，[210， 0,0,0,214]，[310,0,0,113,314]，[410,0,0,213,414]，[610,211,112,313,614]，[710,311,412,613,714]，[810,411,512,713,1014]]

* 注意：
    * board长度将在[3，50]范围内。
    * board[i]长度将在[3，50]范围内。
    * 每个board[i][j]最初都将以[1，2000]范围内的整数开头。

### 解法

就是一道普通模拟题目，对于模拟题目推荐用python这种高度抽象的语言，好写。

先讲讲模拟的思路

1. 首先第一步肯定是消除，直接遍历，然后判断是否有三个格子都是相等的就行了，比如水平的话，若`board[i][j] == board[i][j-1] == board[i][j-2]`，那么这三个格子都需要消除。
    * 因为有水平和垂直**两种消除**，它们可能share同一个格子（考虑L形的一横一竖均为3个格子的情形），因为不能在水平或垂直扫到符合条件的时候就直接将格子置为零。
    * 注意到题目标明格子均为正数，那么我们可以将待消除的格子先变成**相反数**，然后处理完消除再进行一个把所有负数变成零的**零化操作**就行了。
    * 那么，条件判断就变成了`abs(board[i][j]) == abs(board[i][j-1]) == abs(board[i][j-2])`
2. 注意到横消除和竖直消除其实是一种操作，比如你只需要写个横消除，然后横消除后把矩阵**顺时针旋转90度**，再来个横消除，然后再**逆时针旋转回来**就行了。
    * 旋转在python里面是trivial的事情，只需要**一行简单代码**，比如逆时针旋转90度——先`zip(*board)`转置，然后在`[::-1]`颠倒就行了
3. 接下来是把空位（即0的格子）压满，同样只需要写个一个方向的拉，比如下文的`pull_left`，然后旋转跳跃.....
    * 压满就是用`filter`过滤下0，然后再在末端补齐下0保持每行的长度不变就行了。
4. 重复以上过程直到不能再消除。



以下贴出我的解法，主要有几个点需要注意

* 一是步骤比较多，推荐**模块化**，把每个功能抽象成一个函数，即可以增加代码可读性；又方便写完一个函数后即时unit-test该模块的功能是否完整，减少debug难度；还方便代码重复利用。
* 二是......好像也没啥了......

```python
class Solution:
    def candyCrush(self, board: List[List[int]]) -> List[List[int]]:
        if not board or not board[0]:
            return board
        def eliminate(board) -> bool:
            # 水平方向的消除，返回是否有做任意消除
            modified = False
            for i in range(len(board)):
                for j in range(2, len(board[i])):
                    if board[i][j] and
                        abs(board[i][j-2]) == abs(board[i][j-1])
                                            == abs(board[i][j]):
                        board[i][j-2] = board[i][j-1] =
                        board[i][j] = -abs(board[i][j])
                        modified = True
            return modified

        def zeroize(board):
            # 把负数标记给弄成0
            return [[max(i, 0) for i in line] for line in board]

        def rotate(board, clockwise = True):
            # 旋转矩阵
            Tboard = map(list,zip(*board))
            if clockwise:
                return [line[::-1] for line in Tboard]
            else:
                return list(Tboard)[::-1]

        def pullleft(board):
            # 把零视为空格子，往左压
            n_col = len(board[0])
            board = [list(filter(lambda x:x, line)) for line in board]
            board = [line+[0]*(n_col-len(line)) for line in board]
            return board

        modified = True
        while modified:
            modified = False
            modified |= eliminate(board)
            board = rotate(board)
            modified |= eliminate(board)
            board = zeroize(board)
            board = pullleft(board)
            board = rotate(board, False)
        return board
        
```

