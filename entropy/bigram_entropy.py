import jieba
import math
import preprocess_sentence
import time


# 词频统计，方便计算信息熵
def get_tf(tf_dic, words):
    for i in range(len(words)-1):
        tf_dic[words[i]] = tf_dic.get(words[i], 0) + 1


# 二元模型词频统计
def get_bigram_tf(tf_dic, words):
    for i in range(len(words)-1):
        tf_dic[(words[i], words[i+1])] = tf_dic.get((words[i], words[i+1]), 0) + 1


if __name__ == '__main__':
    before = time.time()

    preprocess_sentence.preprocess_sentence()
    with open('./zhwiki_sentence.txt', 'r') as f:
        corpus = []
        count = 0
        for line in f:
            if line != '\n':
                corpus.append(line.strip())
                count += len(line.strip())

    split_words = []
    words_len = 0
    line_count = 0
    words_tf = {}
    bigram_tf = {}

    for line in corpus:
        for x in jieba.cut(line):
            split_words.append(x)
            words_len += 1

        get_tf(words_tf, split_words)
        get_bigram_tf(bigram_tf, split_words)

        split_words = []
        line_count += 1

    print("语料库字数:", count)
    print("分词个数:", words_len)
    print("平均词长:", round(count / words_len, 3))
    # print("语料行数:", line_count)
    # print("非句子末尾词频表:", words_tf)
    # print("二元模型词频表:", bigram_tf)

    bigram_len = sum([dic[1] for dic in bigram_tf.items()])
    print("二元模型长度:", bigram_len)

    entropy = []
    for bi_word in bigram_tf.items():
        jp_xy = bi_word[1] / bigram_len  # 计算联合概率p(x,y)
        cp_xy = bi_word[1] / words_tf[bi_word[0][0]]  # 计算条件概率p(x|y)
        entropy.append(-jp_xy * math.log(cp_xy, 2))  # 计算二元模型的信息熵
    print("基于词的二元模型的中文信息熵为:", round(sum(entropy), 3), "比特/词")  # 6.402

    after = time.time()
    print("运行时间:", round(after-before, 3), "s")
