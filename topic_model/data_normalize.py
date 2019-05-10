# coding:utf-8
"""
@author:hanmy
@file:data_normalize.py
@time:2019/05/10
"""
import glob
import numpy as np
import re
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords


def get_data(path_neg, path_pos):
    neg_data = []
    pos_data = []

    files_neg = glob.glob(path_neg)
    files_pos = glob.glob(path_pos)

    for neg in files_neg:
        with open(neg, 'r', encoding='utf-8') as neg_f:
            neg_data.append(neg_f.readline())

    for pos in files_pos:
        with open(pos, 'r', encoding='utf-8') as pos_f:
            pos_data.append(pos_f.readline())

    neg_label = np.zeros(len(neg_data)).tolist()
    pos_label = np.ones(len(pos_data)).tolist()

    corpus = neg_data + pos_data
    labels = neg_label + pos_label

    return corpus, labels


def normalize(corpus):
    normalized_corpus = []

    for text in corpus:
        # 转为小写字母
        text = text.lower().strip()

        # 去掉符号
        text = re.sub(r"<br />", r" ", text)
        text = re.sub(' +', ' ', text)
        text = re.sub(r'(\W)(?=\1)', '', text)
        text = re.sub(r"([.!?])", r" \1", text)
        text = re.sub(r"[^a-zA-Z.!?]+", r" ", text)

        # 分词并去掉标点符号
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(text)

        # 去掉停用词
        stopword = stopwords.words('english')
        filtered_tokens = [token for token in tokens if token not in stopword]

        # 重新组成字符串
        filtered_text = ' '.join(filtered_tokens)
        normalized_corpus.append(filtered_text)

    return normalized_corpus
