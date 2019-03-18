from gensim.corpora import WikiCorpus
from opencc import OpenCC

openCC = OpenCC('t2s')  # 繁体转简体


# 中文维基百科语料预处理
def preprocess():
    i = 0  # 统计语料中选取的文章篇数
    line = []
    zhwiki_file = './zhwiki-20190301-pages-articles.xml.bz2'  # xml格式的中文维基百科语料，大约1.7GB，选取一部分计算信息熵
    with open('./zhwiki.txt', 'w') as f:  # 将预处理后的语料存入txt格式文件
        wiki = WikiCorpus(zhwiki_file, lemmatize=False, dictionary={})  # 利用gensim的WikiCorpus提取维基语料
        for text in wiki.get_texts():
            for temp_sentence in text:  # 每篇文章一行文本存放
                for x in temp_sentence:  # 去掉所有非中文字符
                    if len(x.encode('utf-8')) == 3:
                        converted = openCC.convert(x)  # 繁体转简体
                        line.append(converted)
            f.write(''.join(line))
            line = []
            i = i + 1

            if i == 10000:
                print("选取中文维基百科的文章篇数:", i)
                break
