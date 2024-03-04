---
title: "Grapn Convolutional Network and Graph Attention Network"
date: 2024-03-03
tags:
  - Graph Neural Network
---

本篇文章介绍图神经网络的两个基础架构Grapn Convolutional Network (GCN) 和 Graph Attention Network (GAT)

## Grapn Convolutional Network

Thomas N. Kipf 和 Max Welling 于2016年提出的GCN架构是图神经网络的开山之作（[paper link](https://arxiv.org/abs/1609.02907)），Max使用GCN架构成功解决了图数据上节点的分类问题。

关于这篇文章及GCN架构，youtube上[这个视频](https://www.youtube.com/watch?v=CwHNUX2GWvE)做了不错的讲解。输入的 $n$ 个节点的图结构$G=(X, A)$，其中$X\in R^{n\times d}$是节点特征矩阵，$A\in R^{n\times n}$是邻接矩阵。其核心的设计为：
$$
H^{(l+1)}=\sigma(D^{-\frac{1}{2}}AD^{-\frac{1}{2}}H^{(l)}W^{(l)})
$$
其中$H^{(l)}$是第 $l$ 次message-passing之后的节点特征矩阵，$H^{(0)}=X$。$W^{(l)}$是第 $l$ 次message-passing对应的模型，可以是一个线性层（Max的文章中是线性层），也可以是更复杂的设计，输入节点特征矩阵$H^{(l)}$，输出一个临时的节点特征矩阵$H^{(l)}_{tmp}$。$D$是一个对角矩阵叫做degree matrix，$D_{ii}$是第 $i$ 个节点的度数（无向图是度数，有向图是入度）

> Max的这篇GCN文章中，包含考虑了self-interaction，也就是说 $A$ 的对角线是单位矩阵 $I$。并且这篇文章并没有考虑边的特征

> 另外，尽管原始论文中喜欢用邻接矩阵运算来方便的表示，并且早期的简单Graph Neural Network也确实可以依赖矩阵运算实现，但是现在我们更偏向另一种更加全面的图表示方法：$G=(X,E,E_{I})$，其中$X\in R^{n\times d_1}$还是初始节点特征矩阵，$E\in R^{m\times d_2}$代表边特征矩阵，边数为$m$，$E_I\in R^{2\times m}$是edge_index矩阵，$E_I[0,j]$和$E_I[1,j]$分别为边 $e_j$ 的起点节点和终点节点

## Graph Attention Network

GCN架构中的message-passing是将每个节点的相邻节点的node_attr做简单平均后作为更新后的node_attr。但是显然我们可以做的更好，比如做一个attention机制的加权平均。

Yoshua Bengio在2018年提出GAT架构（[paper link](https://arxiv.org/abs/1710.10903)）就是这种 idea。[这个视频](https://www.youtube.com/watch?v=iAEDA8aDCZg)对GAT做了较为系统的讲解

其核心设计为：以节点 $i$ 为中心，其一条边两个端点节点的特征 $h_i, h_j$，基于一个attention mechanism $a: R^d\times R^d \rightarrow R$：
$$
e_{ij}=a(Wh_i, Wh_j)
$$
然后对得到的 $e_{ij}$ 进行softmax：
$$
a_{ij}=\frac{\exp(e_{ij})}{\sum\limits_{k\in i的相邻节点}\exp(e_{ik})}
$$
然后 $h_i$ 按 $h_k$们的加权和进行更新：
$$
h_i'=\sigma(\sum\limits_{k\in i的相邻节点}a_{ik}Wh_k)
$$
其中$\sigma()$是非线性activation，$W$是线性层或更复杂的设计

> Bengio在论文中还提及了multi-head attention的设计，这里不详述

## Code Implementation

AIDS是一个抗艾滋药物数据集，输入是药物的结构图（原子成键关系和节点原子种类），输出是二分类结果代表抗艾滋效果。我们以这个数据集为例，基于torch和torch_geometric实现了GCN和GAT。

[repo link](https://github.com/Lingyu-Kong/GraphNeuralNetwork.git)

> torch_geometric是torch上实现图神经网络的一个比较不错的工具包