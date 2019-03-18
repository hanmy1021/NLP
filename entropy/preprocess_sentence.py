from gensim.corpora.wikicorpus import extract_pages
import bz2file  # 通过bz2file不解压读取语料
import re
from opencc import OpenCC

opencc = OpenCC('t2s')


def stop_punctuation(path):  # 中文字符表
    with open(path, 'r') as f:
        return [l.strip() for l in f]


def preprocess_sentence():
    i = 0
    line = ''
    wiki = extract_pages(bz2file.open('./zhwiki-20190301-pages-articles.xml.bz2'))  # 用gensim的extract_pages来提取每个页面
    with open('./zhwiki_sentence.txt', 'w') as f:
        for text in wiki:
            if not re.findall('^[a-zA-Z]+:', text[0]) and text[0] and not re.findall(u'^#', text[1]):  # 去掉帮助页面以及重定向的页面
                converted = opencc.convert(text[1]).strip()  # 繁体转简体
                converted = re.sub('\|\w*\]', '', converted)
                for x in converted:
                    if len(x.encode('utf-8')) == 3 and x not in stop_punctuation('./stop_punctuation.txt'):
                        line += x
                    if x in ['\n', '。', '？', '！', '，', '；', '：'] and line != '\n':  # 以部分中文符号为分割换行
                        f.write(line.strip() + '\n')  # 按行存入语料文件
                        line = ''
                i += 1
            if i == 10:
                print("选取中文维基百科的文章篇数:", i)
                break
