# coding:utf-8
"""
@author:hanmy
@file:hmm_cut.py
@time:2019/04/11
"""


class HMM(object):
    def __init__(self):

        # 状态值集合
        self.state_list = ['B', 'M', 'E', 'S']
        # 状态转移概率
        self.A_dic = {}
        # 观测概率
        self.B_dic = {}
        # 状态的初始概率
        self.Pi_dic = {}
        # 统计状态(B,M,E,S)出现次数，求P(o)
        self.Count_dic = {}

    # 计算转移概率、观测概率以及初始概率
    def train(self, path):

        # 初始化参数
        def init_parameters():
            for state in self.state_list:
                self.A_dic[state] = {s: 0.0 for s in self.state_list}  # 键为状态，值为字典(键为状态，值为转移概率值)
                self.B_dic[state] = {}  # 键为状态，值为字典(键为字或标点，值为观测概率值)
                self.Pi_dic[state] = 0.0
                self.Count_dic[state] = 0

        def makeLabel(text):
            out_text = []
            if len(text) == 1:
                out_text.append('S')
            else:
                out_text += ['B'] + ['M'] * (len(text) - 2) + ['E']
            return out_text

        init_parameters()
        line_num = 0

        with open(path, encoding='utf-8') as f:
            for line in f:
                line_num += 1

                line = line.strip()  # 每个句子去掉首尾空格
                if not line:
                    continue

                word_list = [i for i in line if i != ' ']  # 字的集合
                line_list = line.split()  # 词的集合

                line_state = []  # 句子的状态序列
                for w in line_list:
                    line_state.extend(makeLabel(w))  # 给句子中的每个词打标签

                assert len(word_list) == len(line_state)

                for k, v in enumerate(line_state):
                    self.Count_dic[v] += 1  # 统计每个状态的频数(B,M,E,S)
                    if k == 0:
                        self.Pi_dic[v] += 1  # 每个句子的第一个字的状态(B,S)，用于计算初始状态概率
                    else:
                        self.A_dic[line_state[k - 1]][v] += 1  # 计算转移概率
                        self.B_dic[line_state[k]][word_list[k]] = \
                            self.B_dic[line_state[k]].get(word_list[k], 0) + 1.0  # 计算观测概率

        self.Pi_dic = {k: v * 1.0 / line_num for k, v in self.Pi_dic.items()}
        self.A_dic = {k: {k1: v1 / self.Count_dic[k] for k1, v1 in v.items()} for k, v in self.A_dic.items()}
        self.B_dic = {k: {k1: (v1 + 1) / self.Count_dic[k] for k1, v1 in v.items()} for k, v in
                      self.B_dic.items()}  # 加1平滑

        return self

    def viterbi(self, text, states, start_p, trans_p, emit_p):
        """
        text:要切分的句子
        states:状态值集合(B,M,E,S)
        start_p:初始概率
        trans_p:转移概率
        emit_p:观测概率
        """
        V = [{}]  # 局部概率，每个时刻的概率为{'B':,'M':,'E':,'S':}
        path = {}  # 最优路径

        # 初始化，计算初始时刻所有状态的局部概率，最优路径为各状态本身
        for y in states:
            V[0][y] = start_p[y] * emit_p[y].get(text[0], 0)
            path[y] = [y]
            # 初始化为{'B': ['B'], 'M': ['M'], 'E': ['E'], 'S': ['S']}

        # 递推，递归计算除初始时刻外每个时间点的局部概率和最优路径
        for t in range(1, len(text)):
            V.append({})  # 每个时刻都有一个局部概率，用字典{}表示
            newpath = {}
            # 检验训练的观测概率矩阵中是否有该字
            neverSeen = text[t] not in emit_p['S'].keys() and \
                        text[t] not in emit_p['M'].keys() and \
                        text[t] not in emit_p['E'].keys() and \
                        text[t] not in emit_p['B'].keys()
            for y in states:
                emitP = emit_p[y].get(text[t], 0) if not neverSeen else 1.0  # 如果是未知字，则观测概率同设为1
                (prob, state) = max(
                    [(V[t - 1][y0] * trans_p[y0].get(y, 0) * emitP, y0) for y0 in states if V[t - 1][y0] >= 0])
                # 这里乘观测概率是为了方便计算局部概率，max即记录为局部概率，乘不乘观测概率对于max选择哪个t-1到t的最大概率路径没有影响
                V[t][y] = prob  # 迭代更新t时刻的局部概率V
                newpath[y] = path[state] + [y]  # 实时更新t时刻y状态的最优路径，传统的维特比算法只记录每个时刻的每个状态的反向指针
            path = newpath  # 更新t时刻所有状态的最优路径

        # 终止，观测序列的概率等于T时刻的局部概率
        (prob, state) = max([(V[len(text) - 1][y], y) for y in ['E', 'S']])  # 只需要考虑句尾是E或者S的情况

        return prob, path[state]

    def cut(self, text):
        prob, pos_list = self.viterbi(text, self.state_list, self.Pi_dic, self.A_dic, self.B_dic)  # 返回最优概率和最优路径
        begin = 0
        result = []  # 分词结果
        for i, char in enumerate(text):  # 以字为单位遍历句子
            pos = pos_list[i]
            if pos == 'B':
                begin = i
            elif pos == 'E':
                result.append(text[begin: i + 1])
            elif pos == 'S':
                result.append(char)

        return result


hmm = HMM()
hmm.train('./msr_training.utf8')

text = '这是一个非常棒的方案！'
res = hmm.cut(text)
print(text)
print(res)


def text2tuple(path, cut=True, J=False):
    import jieba
    with open(path) as f:
        dic = {}
        i = 0
        for line in f:
            line = line.strip()
            if cut:
                res = line.split()
            else:
                if J:
                    res = jieba.cut(line)
                else:
                    res = hmm.cut(line)
            dic[i] = []
            num = 0
            for s in res:
                dic[i].append((num, num + len(s) - 1))
                num += len(s)
            i += 1

    return dic


def test(test, gold, J=False):
    dic_test = text2tuple(test, cut=False, J=J)
    dic_gold = text2tuple(gold, J=J)

    linelen = len(dic_test)
    assert len(dic_test) == len(dic_gold)

    num_test = 0
    num_gold = 0
    num_right = 0
    for i in range(linelen):
        seq_test = dic_test[i]
        seq_gold = dic_gold[i]
        num_test += len(seq_test)
        num_gold += len(seq_gold)
        for t in seq_test:
            if t in seq_gold:
                num_right += 1

    P = num_right / num_test
    R = num_right / num_gold
    F1 = P * R / (P + R)
    return P, R, F1


P, R, F1 = test('./msr_test.utf8', './msr_test_gold.utf8')
print("HMM的准确率：", round(P, 3))
print("HMM的召回率：", round(R, 3))
print("HMM的F1值：", round(F1, 3))

P, R, F1 = test('./msr_test.utf8', './msr_test_gold.utf8', J=True)
print("jieba的准确率：", round(P, 3))
print("jieba的召回率：", round(R, 3))
print("jieba的F1值：", round(F1, 3))