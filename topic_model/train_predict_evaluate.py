# coding:utf-8
"""
@author:hanmy
@file:train_predict_evaluate.py
@time:2019/05/10
"""
from sklearn import metrics
import numpy as np


def get_metrics(true_labels, predicted_labels):
    print('精度:', np.round(
        metrics.accuracy_score(true_labels,
                               predicted_labels),
        2))
    print('准确率:', np.round(
        metrics.precision_score(true_labels,
                                predicted_labels,
                                average='weighted'),
        2))
    print('召回率:', np.round(
        metrics.recall_score(true_labels,
                             predicted_labels,
                             average='weighted'),
        2))
    print('F1值:', np.round(
        metrics.f1_score(true_labels,
                         predicted_labels,
                         average='weighted'),
        2))


def train_predict_evaluate_model(classifier,
                                 train_features, train_labels,
                                 test_features, test_labels):
    # 构建模型
    classifier.fit(train_features, train_labels)
    # 在测试集上预测结果
    predictions = classifier.predict(test_features)
    # 评价模型预测表现
    get_metrics(true_labels=test_labels,
                predicted_labels=predictions)

    return predictions
