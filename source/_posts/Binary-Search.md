---
title: All You Need to Know About Binary Search
date: 2020-01-02 19:00:29
tags: Coding
---

Binary search is such an easy to write algorithm but there are usually some hidden annoying bugs, such as condition of **while** statement (easy to get an endless loop). Besides, there are some variants like upper_bound and lower_bound searching, which makes this harder. However, if you can hold on to a same criteria of definition, things will become much easier.

In a nutshell, If you are struggling to write a **CORRECT & BUG-FREE** Binary Search, this might be what you need.

<!--more-->

## Two rules of thumb to remember:

I do not deny that there are other ways of writing binary search, but the following set of criteria is the easiest to understand after I explored, at least for myself.

- Think of `l` and `r` as a left-closed-right-open interval `[l, r)`
- Always use the combination of `while l < r` & `l+1` to avoid endless loop
  - Exit while loop when there is only one element in the interval
  - Endless loop happens when new interval remains the same as the original one **AND** this combination avoids most of the situations like this.

### Ordinary Binary Search

```python
def binary_search(nums, target):
  l, r = 0, len(nums)
  if r==0:
    return -1
  while l < r:
    mid = l + (r - l)//2
    if nums[mid] == target:
      return mid
    elif nums[mid] < target:
      l = mid + 1
    else:
      r = mid
```
This one is the simplest one. It is straightforward and obvious. However, sometimes there are some bugs. Although after some struggling one can always finally figure out what's wrong, the time and energy spent are unnecessary. So next time when you have to write a BS, try to **use the two rules above and do it confidently**!

## Lower Bound & Upper Bound Searching

This is the same definition as `lower_bound` and `upper_bound` function in STL of C++.

- Lower bound:
	- Return index of the first element that does not compare less than the target value
	- which might be index (of the **first** element equal to the target) or (of the **next bigger** element or the end)
- Upper Bound:
	- Return index of the first element that compares greater than the target value
	- which might be index of the **next bigger** element or the end **!!!**

#### Lower Bound Code

```python
def lower_bound(nums, target):
  l, r = 0, len(nums)
  if r == 0:
    return -1
  while l < r:
    mid = l + (r - l)//2
    if nums[mid] >= target:
      r = mid
    else:
      l = mid + 1
  return l
```

- Make thinking easier:
  - Note that although searching interval is `[l, r)`, `r` is also a candidate
  - Think of `target` as a line segment (LS in the following) of equal elements and compare `nums[mid]` to it.
- If `nums[mid]` is to the right of target LS, then the new interval is `[l, mid)` because `nums[mid] `might be the next bigger element
- If `nums[mid]` is on the target LS, which means that `nums[mid]` is equal to target, then the new interval is still `[l, mid)` because `nums[mid]` might be the first element on that LS
- If `nums[mid]` is to the left of target LS, the `mid` must not be the index that we want because what we want is index of a not-less element but this one is smaller. So the new interval is `[mid+1, r)`

#### Upper Bound Code

```python
def upper_bound(nums, target):
  l, r = 0, len(nums)
  if r==0:
    return -1
  while l < r:
    mid = l + (r - l)//2
    if nums[mid] <= target:
      l = mid + 1
    else:
      r = mid
  return l
```

It is almost the same as lower bound code, try to figure out yourself.