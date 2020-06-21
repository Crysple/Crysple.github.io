---
title: Dynamic Programming Over Digits
date: 2020-05-17 13:32:05
tags: Coding
categories: Coding
---

# Introduction

There are many kinds of dynamic programming. **DP over digit**, just as its name shows, is doing dynamic programming over digits of a number. In this post, I will write **a general template** for this kind of problems.

## Problem definition

First of all, we need to figure out what kind of problem it solves. The description of the problem is usually like:

> **Given a interval [lower, upper], find the number of all numbers $i$ that satisfy $f(i)$.**
>
> Here, the condition $f(i)$ is usually irrelated to the size of the number, but the **composition** of this number.

- Sample Problems On Leetcode

  [233. Number of Digit One](https://leetcode.com/problems/number-of-digit-one)
  
  [902. Numbers At Most N Given Digit Set](https://leetcode.com/problems/numbers-at-most-n-given-digit-set)
  
  [1088. Confusing Number II](https://leetcode.com/problems/confusing-number-ii)
  
  [248. Strobogrammatic Number III](https://leetcode.com/problems/strobogrammatic-number-iii)


<!--more-->

Almost all problem is tagged as hard. Fortunately, after you finish reading this post (maybe half an hour), it will only takes you maybe another **half an hour to solve all these four problem**.

# Algorithm for dp over digit

## Overview

- Though I called it dynamic programming above, it is actually searching with memorization. It is using a **top-down searching** from the **first digit** to the **last one** and finally get the **number of solutions** (how many numbers satisfies the condition, etc.), which is the answer.
- So the next problem becomes how to design the searching process. That is, usually the **parameter of the searching function**. Generally, we need to know (though might not all for every problem, just list all here that you can choose a **subset** of them **for a specific problem**.)
  - How to know **which level** we are currently at?  -- using an index variable `idx`, etc.
  - How to know **if it's the first digit**? -- there might be some leading zero
  - How to know **what's the available choices of current digit**?  -- might be limited to `n[idx]`
  - How to know the **previous digits** that we have already collected?
- Then the **prototype of this searching function** might be `dfs(idx, leading, is_limited, prefix)`. In the following I will  explain all these status parameters with an example.

## Details of parameter with an example

### idx and prefix

Simplest parameters:

- **idx**: assuming the given upper bound is `n` (interval might be `[0, n]`,  etc.), we are currently at `n[idx]` after converting n to a string.
- **prefix**: it can mean the previous_digit or all previous digits collected or others depending on the problem

### Leading zeros -- is_leading

Because the number that is being searched may be very long, we usually search from it's most significant digit.

For example, if we would like to search all numbers in `[0, 1000]` whose digits forms an arithmetic sequence with difference as 1 like `345`, `456`, `789`. However, the upper bound of this interval is 1000, which is a four-digit number. This means we need to search from `0000`, then those satisfied number becomes `0345`, `0456`, which are not satisfied now. Hence, we need to add **a flag variable** leading to mark if the **previous digit** is a leading zero. Then,

- If currently **is_leading = True** and **current digit is zero**, then current digit is also a **leading zero**.
- If currently **is_leading = True** but **current digit is not zero**, then current digit act as the most significant digit.

Let the current digit be `d`, then the next function maybe `dfs(idx+1, is_leading && d==0)`

Btw, for this problem, the only thing need to do is only choosing a digit such that `digit==prefix+1` when `is_leading` is False.

### Limitation of current digit -- is_limited

We know that when we are searching, the set of available digits for current digit might change.

For example, when the interval is `[0, 345]`, if the first digit we place is `1`, then the searching range of the second digit is `[0, 9]`. But if the first digit we place is `3`, then the range becomes `[0, 4]`.

To distinguish from these two situations, we introduce the `is_limited` variable. Assuming the current digit is `digit`, upper bound is `n`, then the next searching function is:

- If `is_limited` is 1 and `digit == n[idx]` -- previous digits and current digit are all limited -- then next digit is also limited, which is `dfs(idx+1, is_limited=True)`
- If `is_limited` is 1 but `digit < n[idx]` -- current digit does not reach it's upper bound, then there's no constraint for the range of the next digit.
- If `is_limited` is 0, then the next one is also not limited.

Usally we will first calculate the upper bound of current digit as `limit = n[idx] if is_limit else 9`, which means if it's limited, the upper bound is `n[idx]`, otherwise it's 9. Then in the dfs function, we can have `dfs(idx + 1, is_limited = (digit==limit && is_limited))`

## Template and examples

### [Confusing Numbers II]((https://leetcode.com/problems/confusing-number-ii))

- Idea: search every possible number consisting of confusing digits. Check if it's rotation is the same (then it's not a confused number) when reaches the end.
- Time complexity: $5^{\log n}$, just as other searching solution in discussion.
- This is kind of a bad illustration because it cannot be memorized (every new one is different) -- see the next example for **memorization (so called dp...)**

```python
class Solution:
    def confusingNumberII(self, N: int) -> int:
        sn = str(N)
        # all digits that might causes confuse
        confusing_digits = ['0', '1', '6', '8', '9']
        # rotate mapping, after rotation '6' becomes '9', etc.
        rotate = {'0': '0', '1': '1', '6': '9', '9': '6', '8': '8'}
        def is_confused(number):
            return any(rotate[number[i]] != number[~i] for i in range((len(number)+1)//2))
        #class variable to collect the number of confusing number
        self.ans = 0
 
        def dfs(idx, prefix, is_limited, is_leading):
            '''
            idx: index of current digit in sn
            prefix: just as its name indicates
            is_limited: is current digit limited to N[idx]
            is_leading: mark the first digit (is the previous digit a leading zero)
            '''
      			# upper bound of current digit
            limit = sn[idx] if is_limited else '9'
            if idx == len(sn) - 1:
              	# reach the end, check if the number we've searched is a confusing number
                for digit in confusing_digits:
                    if digit <= limit:
                        self.ans += is_confused(prefix + digit)
                return
           	# otherwise, trying every possible confusing digits
            for digit in confusing_digits:
                if digit > max_digit:
                    break
                dfs(idx+1,
                    prefix+digit if digit != '0' or not is_leading else '',
                    is_limited and digit == sn[idx],
                    is_leading and digit == '0')
        dfs(0, "", True, True)
        return self.ans
```

### [Number of Digit One](https://leetcode.com/problems/number-of-digit-one)

Sometimes you don't need to use all parameter.

- Find the number of `1` in all number between [0, n] inclusively
- The `dfs` return the number of `1`s begining from `idx` with or without limitation.
  - `n_ones` in the function to count `1`s
  - When the current digit is 1, we add `dfs(idx+1, is_limited)` to `n_ones` because every number like `1???` will have `1` as it's leading digit.
  - When the current digit is not 1, we just add `dfs(idx+1, ...)` to `n_ones`
- I use `lru_cache` annotation for memorization, you can also use a 2d matrix to record like `dp[idx][is_limit]` and return directly if it's recorded.
- Time complexity is $9\cdot\log n = O(\log n)$

```python
class Solution:
    def countDigitOne(self, n: int) -> int:
        if n <= 0:
            return 0
        sn = str(n)
        @lru_cache(None)
        def dfs(idx, is_limit):
            if idx == len(sn):
                return 0
            limit = int(sn[idx]) if is_limit else 9
            n_ones = 0
            for digit in range(limit + 1):
                n_ones += dfs(idx+1, is_limit and digit==limit)
                if digit == 1:
                    if idx != len(sn) - 1:
                        n_ones += (int(sn[idx+1:]) + 1) if is_limit and digit==limit else (10 ** (len(sn)-idx-1))
                    else:
                        n_ones += 1
            return n_ones
        return dfs(0, True)
```

- Actually since for all digit's that not 1 and not upper bound, the ans of next recursive dfs is the same, the function can be optimized to 

```python
def dfs(idx, is_limit):
            if idx == len(sn):
                return 0
            limit = int(sn[idx]) if is_limit else 9
            n_ones = 0
            
            if limit >= 1:
                n_ones += (int(sn[idx+1:]) + 1) if limit==1 and idx != len(sn) - 1 else (10 ** (len(sn)-idx-1))
                n_ones += dfs(idx+1, is_limit) + dfs(idx+1, False) * limit
            else:
                n_ones += dfs(idx+1, True)
            return n_ones
```

