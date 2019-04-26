# coding:utf-8
"""
@author:hanmy
@file:ParaEsti.py
@time:2019/04/24
"""
import numpy as np
import matplotlib.pyplot as plt
from gmm import GMM

if __name__ == "__main__":
    N = 10000
    pi = 0.8
    mean_1 = np.array([0.0, 0.0])
    cov_1 = np.mat([[1.0, 0.0], [0.0, 1.0]])
    mean_2 = np.array([3.0, 3.0])
    cov_2 = np.mat([[1.0, 0.5], [0.5, 1.0]])

    gmm = GMM(N, pi, mean_1, cov_1, mean_2, cov_2)
    plt.figure()
    plt.gca().set_aspect('equal')  # 令x轴和y轴的同一区间的刻度相同
    D = gmm.dataset()
    plt.scatter(D[:, 0], D[:, 1])
    plt.show()

    pi_learn, mean_1_learn, cov_1_learn, mean_2_learn, cov_2_learn = gmm.EM(D, N)
    print("权重值：", pi_learn)
    print("第一个分模型的均值：\n", mean_1_learn)
    print("第一个分模型的协方差矩阵：\n", cov_1_learn)
    print("第二个分模型的均值：\n", mean_2_learn)
    print("第二个分模型的协方差矩阵：\n", cov_2_learn)
