---
title: Kernel-Tricks
date: 2019-10-26 19:37:00
tags:
- Machine Learning
mathjax: true
---



Part of the reason why ML algorithms can be so versatile is perhaps the use of **Kernelization**. 

The following shows how to apply kernelization in ridge regression and shows how it can be incorparated in other algorithms.

<!--more-->


## 1. The Overall Idea

### A. Feature expansion

#### i). What it is

Feature expansion is such a **transformation** that maps a original data point (usually expressed as a vector) to another point in another space (another vector).Perhaps an example will make this idea clearly. The following lists two basic feature expansions when there's only two features $(x_1, x_2)$ in a data point.

- Affine expansion $\phi(x) = \phi((x_1, x_2)) \rightarrow (1, x_1, x_2)$
- Quadratic expansion $\phi(x) = \phi((x_1, x_2)) \rightarrow (1, x_1, x_2, x_1^2, x_2^2, x_1, x_2)$

#### ii). Problem: feature dimension explosion

However, if there are a lot of features in a data point, then the dimension of the expanded features vector will become very large. If we generalize the above quadratic expansion to n features, it becomes:

$$\phi(x) = \phi((x_1, x_2, ..., x_n)) \rightarrow (1, x_1, ..., x_n, x_1^2, ..., x_n^2, x_1x_2,...,x_1x_n,...x_{n-1}x_{n})$$

This feature expansion has $1+2n+C_n^2 = \Theta(n^2)$ terms!!! This is really a **feature dimension explosion**. Take MNIST as an example, which has over 700 features. So if we apply quadratic expansion to it, the number of features will become asympotatically $700^2 = 490000$, which is teribble for computing.

#### iii). Solution

Fortunately, there is a trick here. If we **just want to compute the inner product of two transformed vectors**, then things become much easier. Take the quadratic expansion for example. We could prove that

$$\phi(x)^T\phi(x') = (1+x^Tx')^2$$

where x and x' are two data points.

In this way, we could use the original data points to compute the inner product of two transformed vectors without actually transforming them.

So the problem becomes how can we just utilize the inner product of two data points to perform regression model.

### B. Kernel Function

Kernel function denotes an inner product in feature space and is usually denoted as:

$$K(x, x') = \phi(x)^T\phi(x') = <\phi(x), \phi(x')>$$

Using the Kernel function, we could know the inner product of two data points, **without explicitly using $\phi$  to map them into a higher-dimension space**. This is highly desirable, as sometimes our higher-dimensional feature space ($\phi$'s output space) could even be infinite-dimensional and thus unfeasible to compute.

### i). How to take advantage of this

But how can we apply this into our ML algorithms such as ridge regression without explicit mapping data points into the output space of $\phi$ ?



## 2. From Ridge Regression

> The following vectors are all considered as column vectors.

### A. Primary Form of Ridge Regression

We know that by applying normal function, the solution to the ridge regression is usually presented in the following form:

$$\hat w = (A^T A + \lambda I)^{-1}A^Ty$$

where $A = (\phi(x_1)|\phi(x_2)|...|\phi(x_n))^T \in R^{n\times d}$ is the data matrix after feature expansion, I is identity matrix, w is the weight we want to learn and $y = (y_1, y_2, ..., y_n)$ is the true value.

This is called the primary form of Ridge Regression. However, if we wanna apply feature expansion in this, we have to compute $A$ or at least $AA^T$. But we do not want to explicitly apply $\phi$ to the data points. Notice that if we could somehow twist the term $A^T A$ a little bit, say, to $AA^T$, then things become much easier because $(A^TA)_{i, j} = \phi(x_i)^T\phi(x_j) = K(x_i, x_j)$, where $x_i, x_j$ is two data points.

### B. Dual Form of Ridge Regression

#### i). Definition

> We slightly change the definition of A and y for computing convenience.
>
> Now:
>
> $$A = \frac{1}{\sqrt{n}}(\phi(x_1)|\phi(x_2)|...|\phi(x_n))^T \in R^{n\times d}$$
>
> $$y = \frac{1}{\sqrt{n}}(y_1, y_2, ..., y_n)\in R^{n}$$

Thanks to mathematicians, there is sth called linear algebraic identity, which tell us that:

$$(A A^T + \lambda I)^{-1}A^T = A^T(A A^T + \lambda I)^{-1}$$

which can be easily proved if you expand each side

So we have

$$\hat w= (A^T A + \lambda I)^{-1}A^Ty=A^T(A A^T + \lambda I)^{-1}y$$

Denote $\hat \alpha = \frac{1}{\sqrt{n}}(A A^T + \lambda I)^{-1}y$  and $K = AA^T$.

Here K is called  the **Gram Matrix**, which is just the kernel function described in the previous section where $K_{i,j} = \phi(x_i)^T\phi(x_j)$. So we know that until now, we could *compute $\alpha$ without doing actual feature expansion*.

But there is still a $A^T$ in the formula. Does it means we still need to use the $\phi$ function to do features expansion??? The answer is no. We could eliminate that term with the new coming points into a value that can be calculate by the kernel function.

#### ii). Prediction

By the definition above, weight vector becomes:

$$\hat w = A^T(A A^T + \lambda I)^{-1}y = \sqrt{n}A^T\hat \alpha = \sum_{i=1}^n \hat \alpha_i \phi(x_i)$$

When there comes a new point x, we just need to predict by:

$$\phi(x^T)\hat w = \sum_{i=1}^n \hat \alpha_i \phi(x^T) \phi(x_i) = \sum_{i=1}^n \hat \alpha_i K(x^T, x_i)$$

Now we successfully apply kernel function in the ridge regression.

## 3. Conclusion

The Kernel trick is a very interesting and powerful tool. It is powerful because it provides a bridge from linearity to non-linearity to any algorithm that can expressed solely on terms of [dot products](http://en.wikipedia.org/wiki/Dot_product) between two vectors. It comes from the fact that, if we first map our input data into a higher-dimensional space, a linear algorithm operating in this space will behave non-linearly in the original input space.