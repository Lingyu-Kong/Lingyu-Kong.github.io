---
title: '谈谈你没搞清楚的Logit, Logistic Regression'
date: 2024-02-26
tags:
  - Machine Learning
  - Loss Function
---

# 谈谈你没搞清楚的Logit, Logistic Regression


Logistic Regression（以下简称LR）最初是一种用于解决**二分类**问题的损失函数。其名字的Logistic取自[Logistic Distribution](https://en.wikipedia.org/wiki/Logistic_distribution)，这是数学意义上的一种分布类型；而Regression表明LR的损失函数可以将分类问题转化成回归问题来进行解决。

| 你可能听说过，在Machine Learning的早期，人们会将问题大致归为分类问题和回归问题。下面我们会通过数学推导展示其实LR巧妙地将分类问题转化成了回归问题

## Logistic Distribution

关于Logistic Distribution，其实严谨的数学表达式是比较多样的，我们这里给出其中比较基础的概率分布函数：

$$
F(x)=\frac{1}{1+e^{-x}}, -\infin<x<\infin
$$

抛开这个函数作为一个概率分布函数本身的意义，单纯看上面的函数形式，我们会发现这个函数的一个特点在于会将任何的输入x映射到一个$[0,1]$的值域区间。利用这一点，我们可以将一个输出范围为$R$的机器学习模型$f_{\theta}(x)$转化为一个输出为二分类概率的分类模型P

$$
P(y=1|\theta, x)=\frac{1}{1+e^{-f_\theta(x)}}
$$

| 这也就是sigmoid activation 函数

## Logit

logit在数学上指odds的对数，其中odds的定义为一个事件发生和不发生的比率。注意odds不是probability，因为probability的值域为$[0,1]$，而odds的范围为$R$

对于前面我们设计的二分类模型P，其划分为正类的odds可以定义为：

$$
\begin{aligned}
\text{odds}P(y=1|\theta, x)&=\frac{P(y=1|\theta, x)}{P(y=0|\theta, x)} \\
\\
&=e^{f_\theta(x)}
\end{aligned}
$$

那么相应地logit为：

$$
\begin{aligned}
\text{logit}P(y=1|\theta, x)&=\log \text{odds}P(y=1|\theta, x) \\
\\
&=f_\theta(x)
\end{aligned}
$$

很有趣也很令人兴奋，兜兜转转最后logit就是我们之前的模型的输出，也就是说logit最终可以表示成样本特征x的线性回归（Linear Regression）模型，而对于在logit之上套一个sigmoid函数就可以得到概率

## Logistic Regression

首先先放损失函数：

$$
L(y, \hat{y}) = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]
$$

其中$\hat{y}=\text{sigmoid}(f_\theta(x))$

[这篇文章](https://blog.csdn.net/zjuPeco/article/details/77165974)从概率的极大似然估计的角度分析了为什么logistic regression长这个样子

但是只有上面这篇文章不够，因为他没有去解释logistic regression的另外的一个本质：**为什么这个损失函数本质上是一个regression？** 我找到了[另外一篇文章](https://blog.csdn.net/qq_35200479/article/details/94966317)，这篇文章后面的梯度下降部分的推导解释了为什么我们说这本质上是一个regression