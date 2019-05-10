# coding:utf-8
"""
@author:hanmy
@file:classifier.py
@time:2019/05/10
"""
from data_normalize import get_data, normalize
from feature_extractor import bow_extractor, tfidf_extractor
from train_predict_evaluate import train_predict_evaluate_model
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    train_corpus, train_labels = get_data('./train/neg/*.txt', './train/pos/*.txt')
    test_corpus, test_labels = get_data('./test/neg/*.txt', './test/pos/*.txt')

    norm_train_corpus = normalize(train_corpus)
    norm_test_corpus = normalize(test_corpus)

    # 词袋模型特征
    bow_vectorizer, bow_train_features = bow_extractor(norm_train_corpus)
    bow_test_features = bow_vectorizer.transform(norm_test_corpus)

    # tfidf 特征
    tfidf_vectorizer, tfidf_train_features = tfidf_extractor(norm_train_corpus)
    tfidf_test_features = tfidf_vectorizer.transform(norm_test_corpus)

    # 导入分类器
    svm = SGDClassifier(loss='hinge', max_iter=100)
    lr = LogisticRegression(solver='liblinear')

    # 基于词袋模型特征的逻辑斯蒂回归模型
    print("基于词袋模型特征的逻辑斯蒂回归模型")
    lr_bow_predictions = train_predict_evaluate_model(classifier=lr,
                                                      train_features=bow_train_features,
                                                      train_labels=train_labels,
                                                      test_features=bow_test_features,
                                                      test_labels=test_labels)

    # 基于词袋模型的支持向量机模型
    print("基于词袋模型的支持向量机模型")
    svm_bow_predictions = train_predict_evaluate_model(classifier=svm,
                                                       train_features=bow_train_features,
                                                       train_labels=train_labels,
                                                       test_features=bow_test_features,
                                                       test_labels=test_labels)

    # 基于tfidf的逻辑斯蒂回归模型
    print("基于tfidf的逻辑斯蒂回归模型")
    lr_tfidf_predictions = train_predict_evaluate_model(classifier=lr,
                                                        train_features=tfidf_train_features,
                                                        train_labels=train_labels,
                                                        test_features=tfidf_test_features,
                                                        test_labels=test_labels)

    # 基于tfidf的支持向量机模型
    print("基于tfidf的支持向量机模型")
    svm_tfidf_predictions = train_predict_evaluate_model(classifier=svm,
                                                         train_features=tfidf_train_features,
                                                         train_labels=train_labels,
                                                         test_features=tfidf_test_features,
                                                         test_labels=test_labels)