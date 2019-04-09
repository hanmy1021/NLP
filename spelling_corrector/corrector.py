# coding:utf-8
"""
@author:hanmy
@file:corrector.py
@time:2019/04/05
"""

import re
import collections


# 返回单词的列表
def words(text):
    return re.findall('[a-z]+', text.lower())


# 统计每个单词出现的次数，用于计算每个单词出现的频率，平滑处理默认词频为1
def train(words):
    model = collections.defaultdict(lambda: 1)
    for word in words:
        model[word] += 1
    return model


# 读取整个文本，并且把文本中的单词进行统计
with open('./big.txt') as f:
    words_tf = train(words(f.read()))

# 小写字母列表
alphabet = 'abcdefghijklmnopqrstuvwxyz'


# 计算每个与给定单词编辑距离为1的单词，并组成一个集合
def edits(word):
    # 将每个单词都分割为两两一组，方便后边的编辑距离的计算
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    # 删除的编辑距离：每次从b字符串中删除一个字符
    deletion = [a + b[1:] for a, b in splits if b]
    # 替换的编辑距离：每次替换b中的一个字母
    substitution = [a + c + b[1:] for a, b in splits for c in alphabet if b]
    # 插入的编辑距离：向每个单词分割的组插入一个字母
    insertion = [a + c + b for a, b in splits for c in alphabet]
    return set(deletion + substitution + insertion)


# 计算单词列表words中，在文件中的单词的列表
def known(words):
    return set(w for w in words if w in words_tf)


# 计算候选单词中词频最大的单词
def correct(word):
    candidates = known([word]) or known(edits(word)) or [word]
    return max(candidates, key=words_tf.get)


print(correct('speling'))
print(correct('cerrect'))
