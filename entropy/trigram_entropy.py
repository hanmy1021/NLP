import jieba
import math
import preprocess_sentence
import time


# 非句子末尾二元词频统计
def get_bi_tf(tf_dic, words):
    for i in range(len(words)-2):
        tf_dic[(words[i], words[i+1])] = tf_dic.get((words[i], words[i+1]), 0) + 1


# 三元模型词频统计
def get_trigram_tf(tf_dic, words):
    for i in range(len(words)-2):
        tf_dic[((words[i], words[i+1]), words[i+2])] = tf_dic.get(((words[i], words[i+1]), words[i+2]), 0) + 1


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
    trigram_tf = {}

    for line in corpus:
        for x in jieba.cut(line):
            split_words.append(x)
            words_len += 1

        get_bi_tf(words_tf, split_words)
        get_trigram_tf(trigram_tf, split_words)

        split_words = []
        line_count += 1

    print("语料库字数:", count)
    print("分词个数:", words_len)
    print("平均词长:", round(count / words_len, 3))
    # print("语料行数:", line_count)
    # print("非句子末尾二元词频表:", words_tf)
    # print("三元模型词频表:", trigram_tf)

    trigram_len = sum([dic[1] for dic in trigram_tf.items()])
    print("三元模型长度:", trigram_len)

    entropy = []
    for tri_word in trigram_tf.items():
        jp_xy = tri_word[1] / trigram_len  # 计算联合概率p(x,y)
        cp_xy = tri_word[1] / words_tf[tri_word[0][0]]  # 计算条件概率p(x|y)
        entropy.append(-jp_xy * math.log(cp_xy, 2))  # 计算三元模型的信息熵
    print("基于词的三元模型的中文信息熵为:", round(sum(entropy), 3), "比特/词")  # 0.936

    after = time.time()
    print("运行时间:", round(after-before, 3), "s")
