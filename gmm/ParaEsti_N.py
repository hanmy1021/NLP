# coding:utf-8
"""
@author:hanmy
@file:ParaEsti_N.py
@time:2019/04/24
"""
import numpy as np
import matplotlib.pyplot as plt
from gmm import GMM


# 计算steps次参数估计的权重值的均值和方差
def pi_N(N, pi, mean_1, cov_1, mean_2, cov_2, steps):
    gmm = GMM(N, pi, mean_1, cov_1, mean_2, cov_2)
    pi_steps = np.zeros(shape=steps)
    # 均值
    pi_mu = 0
    # 方差
    pi_sigma = 0

    # 计算steps次估计值和均值
    for i in range(steps):
        D = gmm.dataset()
        pi_learn, _, _, _, _ = gmm.EM(D, N)
        pi_mu += pi_learn
        pi_steps[i] = pi_learn

    pi_mu /= steps

    # 计算steps次方差
    for i in range(steps):
        pi_sigma += (pi_steps[i]-pi_mu)**2

    pi_sigma /= steps

    return pi_mu, pi_sigma


if __name__ == "__main__":
    # 样本点数Ni
    N1 = 100
    N2 = 1000
    N3 = 10000

    pi = 0.8
    mean_1 = np.array([0.0, 0.0])
    cov_1 = np.mat([[1.0, 0.0], [0.0, 1.0]])
    mean_2 = np.array([3.0, 3.0])
    cov_2 = np.mat([[1.0, 0.5], [0.5, 1.0]])

    # 参数估计次数steps
    steps = 10

    Y_mu = []
    Y_sigma = []

    pi_mu_100, pi_sigma_100 = pi_N(N1, pi, mean_1, cov_1, mean_2, cov_2, steps)
    Y_mu.append(pi_mu_100)
    Y_sigma.append(pi_sigma_100)
    print("N=100时学习", steps, "次得到的权重值均值：", pi_mu_100, ",方差：", pi_sigma_100)

    pi_mu_1000, pi_sigma_1000 = pi_N(N2, pi, mean_1, cov_1, mean_2, cov_2, steps)
    Y_mu.append(pi_mu_1000)
    Y_sigma.append(pi_sigma_1000)
    print("N=1000时学习", steps, "次得到的权重值均值：", pi_mu_1000, ",方差：", pi_sigma_1000)

    pi_mu_10000, pi_sigma_10000 = pi_N(N3, pi, mean_1, cov_1, mean_2, cov_2, steps)
    Y_mu.append(pi_mu_10000)
    Y_sigma.append(pi_sigma_10000)
    print("N=10000时学习", steps, "次得到的权重值均值：", pi_mu_10000, ",方差：", pi_sigma_10000)

    Y_mu.append(0.8)
    X = [1, 2, 3, 4]
    plt.bar(X, Y_mu, align='center', tick_label=['N=100', 'N=1000', 'N=10000', 'standard value'], width=0.5)
    plt.ylim((0.7, 0.85))
    plt.xlabel('different N')
    plt.ylabel('mean of pi')
    plt.show()
