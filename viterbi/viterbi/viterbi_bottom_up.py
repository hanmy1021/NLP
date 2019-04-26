# coding:utf-8
"""
@author:hanmy
@file:Viterbi.py
@time:2019/04/10
"""

import numpy as np
M = 2
N = 3
A = np.array([[0.5, 0.2, 0.3],
             [0.3, 0.5, 0.2],
             [0.2, 0.3, 0.5]])
B = np.array([[0.5, 0.5],
             [0.4, 0.6],
             [0.7, 0.3]])
pi = np.array([0.2, 0.4, 0.4])
T = 3
O = [1, 2, 1]
'''
N为隐藏状态数目
M为观测符号数目
A为状态转移矩阵：N×N
B为观测矩阵：N×M
pi为初始向量：1×N
T为观测符号序列长度
O为观测序列
'''


def Viterbi(M, N, A, B, pi, T, O):
    delta = np.zeros(shape=(T, N))  # delta为局部概率，即到达某个特殊的中间状态时的概率
    psi = {}  # psi为反向指针，指向最优的引发当前状态的前一时刻的某个状态

    # 初始化，计算初始时刻所有状态的局部概率，反向指针均为0
    # delta[0] = pi * B[:, O[1]-1]
    for i in range(N):
        delta[0][i] = pi[i] * B[i][O[0] - 1]
        psi[i] = [i]

    # 递推，递归计算除初始时刻外每个时间点的局部概率和反向指针
    for t in range(1, T):
        path = {}
        for j in range(N):
            maxval, ind = max([(delta[t - 1][k] * A[k][j], k) for k in range(N)])
            # 找到从t-1时刻到t时刻第j个状态的最大概率和使从t-1时刻到t时刻第j个状态的概率为最大的t-1时刻的状态序号
            delta[t][j] = maxval * B[j][O[t] - 1]  # 更新局部概率
            path[j] = psi[ind] + [j]  # 更新t时刻状态j的最优路径
        psi = path

    # 终止，观测序列的概率等于T时刻的局部概率
    p, q = max([(delta[T - 1][k], k) for k in range(N)])  # 输出最优概率和第T时刻哪个状态(q)达到最优概率
    psi[q] = [psi[q][i] + 1 for i in range(N)]  # 输出T时刻状态q的最优路径，+1是为了输出状态序号1~3，而不是0~2

    return psi[q], p, delta


q, p, delta = Viterbi(M, N, A, B, pi, T, O)
print("最优状态序列：", q)
print("最优路径概率：", p)
print("局部概率矩阵：", delta)
