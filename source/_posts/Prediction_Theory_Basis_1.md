---
title: Basics of Prediction Theory 1（预测理论基础一）
date: 2019-12-25 22:31:31
tags: Machine Learning

---

如今人工智能盛行当道，主要得益于近年机器学习和神经网络的大力发展，而这些发展实际上都离不开数学的理论支撑。在这些数学内容里面，最重要的莫过于概率论和线性代数，预测理论属于概率论数理统计的一部分，掌握好理论基础对于后续概念的学习和理解十分重要。

这里简单记录下自己学习的机器学习相关的预测理论部分内容。特别的，本文是关于如何给无输入且只有两个可能结果的事件建模，做出预测和评估，下一篇则泛化到如何给标记数据（即每个数据点包括输入x和输出标记y）建模，预测和评估。

<!--more-->

## 统计模型之理想状态

如果要你预测一枚抛出去的硬币落地之后是头朝上还是尾朝上，你会怎么做呢？

直觉上，如果这枚硬币是“公平”的，具备质量绝对均匀等条件，那么我们知道预测哪一面都一样，但是，如果这枚硬币本身就是有偏差的呢？这时我们就必须选择一面“好过”另外一面的来进行预测。

一般来说，为了预测抛出的硬币最终哪一面朝上，人们有两种方案选择，物理建模和统计建模。

* *物理模型*：这需要你各种初始条件和一堆物理知识去进行受力分析，精确地建模出硬币的运行轨迹。 $\rightarrow$ 难度过大
* *统计模型*：另一方面，人们也可以借助一个统计模型来进行对未知结果的预测，然后评估不同预测策略的优劣。在抛硬币这件事情上，如果我们将（硬币哪面朝上这个）结果视为一个随机变量，那么两个结果就各自有一个概率，假定头朝上的概率为$\theta$，那么尾朝上的概率就是$1-\theta$；同时，我们将这个随机变量记作Y，且Y=1代表头朝上，Y=0代表尾朝上。在这里，$\theta$，就是这个统计模型的参数，它可以取值的范围是[0, 1]，也叫作模型的参数空间。注意到，这个模型其实就是伯努利分布的一种，记作Y ~ Bern($\theta$)。

那么拥有了统计模型之后，我们是怎么预测结果的呢？以上文的伯努利分布Bern($\theta$)为例，有两种情况，一是我们知道模型参数$\theta$，二是不知道，这一小节我们先讨论在知道参数的前提下，如何作出预测，其中一种可能的预测如下：

1. 如果$\theta>\frac{1}{2}$，即头朝上的概率大，那么我们一直预测头朝上就行了
2. 反过来，若$\theta<\frac{1}{2}$，那么我们预测尾朝上

如果使用这个策略，通过简单计算我们可以得知，它的**错误率**是$\min \{\theta, 1-\theta\}$，其实，这也是最优的预测策略。

但是，上面的策略是建立在模型参数$\theta$已知的前提下。然而一般来说对于我们想预测的事件，我们并不知道建立统计模型后，模型的参数是啥。还是以抛硬币为例，$\theta$可能受多方面因素的影响，比方说硬币的质量密度，是头部的质量密度大还是尾部等等。以下我们着重讨论这种情况。

## 统计模型之现实情况

在上面的例子中，我们用Bern($\theta$)去对抛硬币这件事情建模，如果我们知道参数$\theta$，那么我们可以直接采用上述的分布，但一般来讲我们不会知道这个参数的具体值。幸运的是，我们可以通过这个模型分布产生的数据估计它。

### 插入原则（Plug-in principle）

现在我们尝试用比较正式的语言来描述下这件事情：假设知道抛这枚硬币的前n次的结果$Y_1, ..., Y_n$（原模型分布产生数据），那么目标就是用这些数据来对第n+1次的抛硬币结果$Y$进行预测。注意到$Y_1, ..., Y_n, Y$都是iid，即独立同分布的。一般我们写作：
$$Y_1, ..., Y_n, Y \text{~ iid } P$$
P就是那个我们不知道的产生这些结果的原分布啦。总结以下，Plug-in principle就是用

1. *原模型分布产生的数据的分布*去估算*模型本身真正的分布*。
2. 然后将我们*预估出来的参数插入到原分布*以帮助我们做出*最优预测*。

那么什么时候我们可以运用这个原则去进行估算呢？有两个条件：

1. 观察到的数据必须跟我们想预测的结果相关
2. 观察到的数据和结果必须都是iid随机变量

说句题外话，iid这个条件在Machine Learning真是无处不在，因为它在可观察的数据和我们要预测的目标之间建立了一种简单的联系。

### 最大似然估计（Maximum likelihood estimation）

在Plug-in principle中，我们需要用观察到的数据去预估原模型分布的参数，最常用的方法就是MLE了。上过概率论的朋友想必都知道MLE，这里就将上文的Bern($\theta$)继续往下推演。

简单来说，MLE就是先把未知的参数当作是固定的变量，用假定的分布$P_{\theta}$计算其产生所观察到的所有数据$(y_1, ..., y_n)$的联合分布概率$P_{\theta}(y_1, ..., y_n)$（这里的概率被赋予另外一个名字叫likelihood，我们记为对$\theta$的函数$L(\theta)$），然后通过对参数求导，置为零，计算得能最大化该概率的参数。一般来说我们假定各个观察到的数据均为idd，即独立不相关，故
$$L(\theta)=P_{\theta}(y_1, ..., y_n) = \prod_{i=1}^n P_{\theta}(y_i)$$
又因为$0<P_{\theta}(y_i)<1$，多个变量相乘可能导致数值变得过小，同时我们只是想最大化它，故可以套多一层log将累乘变成累加的形式，得到
$$\ln L(\theta) = \ln P_{\theta}(y_1, ..., y_n) = \sum_i^n \ln P_{\theta}(y_1)$$
这个也被称之为log-likelihood.

> #### Probability vs Likelihood
>
> * Wiki上对likelihood的定义为：The number that is the probability of some observed outcomes given a set of parameter values is regarded as the **likelihood of the set of parameter values given the observed outcomes**.
> * 意思就是，给定一些分布产生的数据，关于这个分布某个 *参数集合* 的likelihood是这些数据的联合概率分布。
> * 那么和probability又有什么区别呢？简单比较就是，probability是用来衡量出现某个结果的可能性的，而likelihood是衡量某个假设（即我们假设该分布的参数为某个固定值$\theta$）的可能性的。

在上边的例子中，假设头朝上为正例，我们观察到的数据$y_i=1$，反之则$y_i=0$，那么log-likelihood为

$$\ln L(\theta) = \ln \prod_{i=1}^n \theta^y_i (1-\theta)^{1-y_i} = \sum_{i=1}^n y_i\ln\theta (1-y_i)\ln (1-\theta)$$

令$\frac{d\ln L(\theta)}{d\theta} = \frac{1}{\theta}\sum_{i=1}^n y_i -\frac{1}{1-\theta}\sum_{i=1}^n (1-y_i) = 0$，解得 $\theta = \frac{\sum_{i=1}^n y_i}{n}$

接着我们求二次导检查下它是一个最小值，最大值还是鞍点如下：

$$\frac{d^2\ln L(\theta)}{d\theta^2} = -\frac{1}{\theta^2}\sum_{i=1}^n y_i -\frac{1}{(1-\theta)^2}\sum_{i=1}^n (1-y_i)$$

显然，当$\theta\in [0,1]$，它大于等于零，所以这是一个最大值。

因此，$\hat \theta = \frac{\sum_{i=1}^n y_i}{n}=\bar y$，这里的$\hat \theta$上的标志代表这是我们预估出来的参数，而非原分布实际的参数$\theta$。

### 插入预测的测试错误率

紧接着，我们就可以将预估出来的参数plug in原分布，进行预测了，因为哪头的概率大，我们就预测哪头。注意到，分布参数是基于$(y_1, ..., y_n)$计算得出的，所以我们将预测结果记为
$$\hat y(y_1, ..., y_n) = \unicode{x1D7D9}_{\{\hat \theta(y_1, ..., y_n)>1/2\}}$$
注意到这里的y也加了上标，代表这是我们预估出来的值，也叫做plug-in prediction。而$\unicode{x1D7D9}$被称为 Indicator function，当其下标条件为真时其值为1，为假时其值为0。

那么其错误率是多少呢？在plug-in principle小节谈到的iid模型中，预测值$\hat Y=\hat y(Y_1, ..., Y_n)$不等于真实值$Y$的概率可以这么计算：

$$P(\hat Y \ne Y) = P(\hat y(Y_1, ..., Y_n) \ne Y) = P(\frac{\sum_{i=1}^n Y_i}{n}>\frac{1}{2})\cdot P(Y=0)+ P(\frac{\sum_{i=1}^n Y_i}{n}\le\frac{1}{2})\cdot P(Y=1)$$

假设在 Y ~ Bern$(\theta)$中，$\theta > 1/2$，我们可以用一个伯努利分布随机变量的尾部概率上限来计算它的边界如下：
$$
\begin{aligned}P(\hat Y \ne Y) &= (1-P(\frac{\sum_{i=1}^nY_i}{n}\le\frac{1}{2}))\cdot(1-\theta)+P(\frac{\sum_{i=1}^nY_i}{n}\le\frac{1}{2})\cdot \theta \\&= (1-\theta)+(2\theta-1)\cdot P(\frac{\sum_{i=1}^nY_i}{n}\le\frac{1}{2}) \\&\le (1-\theta) + (2\theta - 1) \cdot e^{-n\cdot RE(1/2, p)}\end{aligned}
$$
这里的RE是用于两个伯努利分布之间Bern(a) & Bern(b)之间的相对熵[Relative Entropy](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)函数（也可理解为某种距离），它的公式是
$$
RE(a, b) = a \ln \frac{a}{b} + (1-a) \ln \frac{1-a}{1-b}
$$
同样的，当$\theta \le 1/2$ 的时候，我们可以得到
$$
\begin{aligned}P(\hat Y \ne Y) &\le \theta + (1-2\theta) \cdot e^{-n\cdot RE(1/2, p)}\end{aligned}
$$
整合起来，得到
$$
P(\hat Y \ne Y) \le \min\{\theta, 1-\theta\} + |2\theta - 1|\cdot e^{-n\cdot RE(1/2, \theta)}
$$
注意到在最优预测（即知道原参数$\theta$的情况下），我们预测错误的概率是$\{\theta, 1-\theta\}$。这里，因为相对熵RE总是非负的，并且当且仅当a = b时候，RE(a, b) = 0。所以，上面这个错误的概率只是比最优预测多出一个很小的量，且当n变大的时候，这个量以指数速度趋向于零。