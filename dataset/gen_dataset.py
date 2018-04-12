import os
import jieba
import random

_current_dir = os.path.dirname(__file__)
_new_hans_dict = _current_dir + '/' + 'new_hans.txt'

patterns = (
    "<{keyword}>"
    "帮我找一下<{keyword}>",
    "查询一下<{keyword}>",
    "查询<{keyword}>",
    "搜索<{keyword}>",
    "搜寻<{keyword}>",
    "搜索一下<{keyword}>",
    "搜索课程<{keyword}>",
    "查找课程<{keyword}>",
    "搜索一下<{keyword}>",
    "查一下<{keyword}>",
    "有没有<{keyword}>",
    "找一下<{keyword}>",
    "找找<{keyword}>",
    "能不能找到<{keyword}>",
    "能不能搜到<{keyword}>",
)

headers = (
    "请问",
    "你好",
    "Hi",
    "嗨",
    "喂",
    "问一下",
    "",
)

tailers = (
    "",
    "好吗",
    "可以吗",
)


def read_keywords():
    with open(_new_hans_dict, 'r') as f:
        for line in f:
            try:
                word, tag, *_ = line.split('\t')
                if tag and tag[0] in ('n', 'i', 'v', 'g'):
                    yield word
            except:
                pass


def tagging(words):
    next_mark = 0
    for w in words:
        if w == '<':
            next_mark = 1
        elif w == '>':
            next_mark = 0
        else:
            yield w, next_mark


def generate_dataset():
    ws = list(read_keywords())
    random.shuffle(ws)
    ws = ws[:50000]
    # ws = ws[:10]
    for p in patterns:
        for h in headers:
            for t in tailers:
                for w in ws:
                    sentence = h + p.format(keyword=w) + t
                    words = list(jieba.cut(sentence))
                    tags = list(tagging(words))
                    words = [w for w, _ in tags]
                    tags = [tag for _, tag in tags]
                    yield words, tags
