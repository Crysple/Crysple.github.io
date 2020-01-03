---
title: A Intuitive Deduction of Multinomial Distribution
date: 2018-02-29
tags:
- Math
- NLP
mathjax: true

---

因为之前看了一篇关于Topic word的论文，里面谈到主题模型LDA，故想仔细了解下这个模型的方方面面，在看文献的同时发现之前学的概率论已经零零碎碎，所以对几个相关的概率分布做一个回顾及再学习

这篇是关于多项分布的内容，也是笔记吧，因为做笔记的时候用的是英文，就懒得二次翻译了
个人觉得从二项分布开始，更能直观地了解多项分布的意义以及很容易地记住公式

<!--more-->

---

## Multinomial Distribution


### For **Binomial Distribution**

In probability theory and statistics, the binomial distribution with parameters n and p is the discrete probability distribution of the number of successes in a sequence of n independent experiments, each asking a yes–no question, and each with its own boolean-valued outcome: a random variable containing single bit of information: success/yes/true/one (with probability p) or failure/no/false/zero (with probability q = 1 − p). - From wikipedia

- The simplest case is flipping a coin:
	- Let p be the possibility of the upper side is a specific side of the coin
	- Then the possibility that flip the coin n times and every time the upper side is the assigned side of the coin, conforms the binomial distribution

In general, if the random variable X follows the binomial distribution with parameters $n \in ℕ$ and $p \in [0,1]$, we write $X \sim B(n, p)$. The probability of getting exactly k successes in n trials is given by the probability mass function:

${\displaystyle Pr(k;n,p)=\Pr(X=k)={n \choose k}p^{k}(1-p)^{n-k}}$
for $k = 0, 1, 2, ..., n$, where $${\binom {n}{k}}={\frac {n!}{k!(n-k)!}}$$

### Another representation

- The probability of getting successes is $p_1$, whose time is $k_1$
- The probability of getting faulure is $p_2$, whose time is $k_2$
- $p_1+p_2=1$
- Thus, $$\Pr(k_1,k_2,p_1,p_2) ={\binom {n}{k}}={\frac {n!}{k_1!k_2!}}p_1^{k_1}p_2^{k_2}$$

### Multinomial Distribution

Extend the above situation, we get the multinomial distribution:

- Let’s say, the experiment:
	- It has k kinds of possible results, whose possibilities are respectively $p_1,p_2,…,p_k$
	- Repeat the experiment n times
	- The times that every possible result occurs are respectively $x_1,x_2,…,x_k$

- Then, we have:

$${\begin{aligned}f(x_{1},\ldots ,x_{k};n,p_{1},\ldots ,p_{k})&{}=\Pr(X_{1}=x_{1}{\text{ and }}\dots {\text{ and }}X_{k}=x_{k})\\&{}={\begin{cases}{ {n! \over x_{1}!\cdots x_{k}!}p_{1}^{x_{1}}\times \cdots \times p_{k}^{x_{k}}},\quad &{\text{when }}\sum _{i=1}^{k}x_{i}=n\\\\
0&{\text{otherwise,}}\end{cases}}\end{aligned}}$$
for non-negative integers x1, ..., xk.