# coding:utf-8
"""
@author:hanmy
@file:gmm.py
@time:2019/04/24
"""
import numpy as np


class GMM:
    def __init__(self, N, pi, mean_1, cov_1, mean_2, cov_2):
        # 采样点数
        self.N = N
        # 高斯分布的权重
        self.pi = pi
        # 第一个二维高斯分布的均值
        self.mean_1 = mean_1
        # 第一个二维高斯分布的协方差矩阵
        self.cov_1 = cov_1
        # 第二个二维高斯分布的均值
        self.mean_2 = mean_2
        # 第二个二维高斯分布的协方差矩阵
        self.cov_2 = cov_2

    # 生成观测数据集
    def dataset(self):
        # 数据集大小为(N,2)
        D = np.zeros(shape=(self.N, 2))
        for i in range(self.N):
            # 产生0-1之间的随机数
            j = np.random.random()
            if j < self.pi:
                # pi的概率以第一个二维高斯分布产生一个样本点
                x = np.random.multivariate_normal(mean=self.mean_1, cov=self.cov_1, size=1)
            else:
                # 1-pi的概率以第二个二维高斯分布产生一个样本点
                x = np.random.multivariate_normal(mean=self.mean_2, cov=self.cov_2, size=1)
            D[i] = x
        return D

    def EM(self, D, N):
        # 高斯分布密度phi
        def phi(x, mean, cov):
            return np.exp(-(x - mean) * np.linalg.pinv(cov) * (x - mean).T / 2) / (
                        2 * np.pi * np.sqrt(np.linalg.det(cov)))

        # 分模型对观测数据的响应度gamma
        def gamma(D, j, k, p, m_1, c_1, m_2, c_2):
            # 第一个分模型
            if k == 0:
                return p * phi(D[j], m_1, c_1) / (p * phi(D[j], m_1, c_1) + (1 - p) * phi(D[j], m_2, c_2))
            # 第二个分模型
            elif k == 1:
                return (1 - p) * phi(D[j], m_2, c_2) / (p * phi(D[j], m_1, c_1) + (1 - p) * phi(D[j], m_2, c_2))

        # 模型的均值mu
        def mu(D, g, N):
            # 第一个分模型的均值的分子部分
            mu_1a = np.array([[0.0, 0.0]])
            # 第一个分模型的均值的分母部分
            mu_1b = np.array([[0.0, 0.0]])
            # 第二个分模型的均值的分子部分
            mu_2a = np.array([[0.0, 0.0]])
            # 第二个分模型的均值的分母部分
            mu_2b = np.array([[0.0, 0.0]])
            for j in range(N):
                mu_1a += g[j][0] * D[j]
                mu_1b += g[j][0]
                mu_2a += g[j][1] * D[j]
                mu_2b += g[j][1]
            # 返回第一个分模型的均值和第二个分模型的均值，都是一行两列的矩阵
            return mu_1a / mu_1b, mu_2a / mu_2b

        # 模型的协方差矩阵sigma
        def sigma(D, m1, m2, g, N):
            # 第一个分模型的协方差矩阵的分子部分
            sigma_1a = np.mat([[0.0, 0.0], [0.0, 0.0]])
            # 第一个分模型的协方差矩阵的分母部分
            sigma_1b = np.mat([[0.0, 0.0], [0.0, 0.0]])
            # 第二个分模型的协方差矩阵的分子部分
            sigma_2a = np.mat([[0.0, 0.0], [0.0, 0.0]])
            # 第二个分模型的协方差矩阵的分母部分
            sigma_2b = np.mat([[0.0, 0.0], [0.0, 0.0]])
            for j in range(N):
                sigma_1a += g[j][0] * ((D[j] - m1).T * (D[j] - m1))
                sigma_1b += g[j][0]
                sigma_2a += g[j][1] * ((D[j] - m2).T * (D[j] - m2))
                sigma_2b += g[j][1]
            # 返回第一个分模型的协方差矩阵和第二个分模型的协方差矩阵，都是两行两列的矩阵
            return sigma_1a / sigma_1b, sigma_2a / sigma_2b

        # 模型的权重alpha
        def alpha(g, N):
            a = 0
            for j in range(N):
                # 只需要求第一个分模型的权重
                a += g[j][0]
            return a / N

        p = 0.5
        m_1 = np.array([[0.0, 0.0]])
        c_1 = np.mat([[1.0, 0.0], [0.0, 1.0]])
        m_2 = np.array([[1.0, 1.0]])
        c_2 = np.mat([[1.0, 0.0], [0.0, 1.0]])

        # 权重pi的变化量初始化为1
        delta_p = 1
        # 迭代次数i
        i = 0
        # 当权重pi的变化量小于10e-7时停止迭代
        while delta_p > 10e-7:
            Gamma = np.zeros(shape=(N, 2))
            for j in range(N):
                for k in range(2):
                    # 计算分模型k对观测数据的响应度gamma
                    Gamma[j][k] = gamma(D, j, k, p, m_1, c_1, m_2, c_2)

            # 更新模型的均值
            m_1, m_2 = mu(D, Gamma, N)
            # 更新模型的协方差矩阵
            c_1, c_2 = sigma(D, m_1, m_2, Gamma, N)
            # 计算权重pi的变化量
            delta_p = abs(p - alpha(Gamma, N))
            # 更新模型的权重
            p = alpha(Gamma, N)

            '''
            i += 1
            # 每五次迭代打印权重值
            if i % 5 == 0:
                print(i, "steps' pi:", p)
            '''

        # 返回EM算法学习到的五个参数
        return p, m_1, c_1, m_2, c_2
