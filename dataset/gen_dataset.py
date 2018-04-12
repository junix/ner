import os
import jieba
import random

_current_dir = os.path.dirname(__file__)
_new_hans_dict = _current_dir + '/' + 'new_hans.txt'

patterns = (
    "<{keyword}>",
    "<{keyword}>",
    "<{keyword}>",
    "找一下<{keyword}>",
    "查一下<{keyword}>",
    "搜一下<{keyword}>",
    "查询一下<{keyword}>",
    "查询<{keyword}>",
    "搜索<{keyword}>",
    "搜寻<{keyword}>",
    "搜索一下<{keyword}>",
    "搜索课程<{keyword}>",
    "查找课程<{keyword}>",
    "搜索一下<{keyword}>",
    "有没有<{keyword}>",
    "有<{keyword}>",
    "找找<{keyword}>",
    "找到<{keyword}>",
    "搜到<{keyword}>",
)

headers = (
    "请问",
    "麻烦问一下",
    "你好",
    "Hi,",
    "嗨,",
    "喂,",
    "问一下",
    "麻烦你",
    "",
)

seconds = (
    "能不能",
    "可不可以",
    "可否",
    "能否",
    "是否",
    "帮我",
    "",
)

tailers = (
    "",
    ",好吗",
    ",可以吗",
    "好吗",
    "相关的"
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


def generate_dataset(size=200000):
    ws = list(read_keywords())
    count = 0
    while count < size:
        w = random.choice(ws)
        h = random.choice(headers)
        s = random.choice(seconds)
        p = random.choice(patterns)
        t = random.choice(tailers)
        sentence = h + s + p.format(keyword=w) + t
        words = list(jieba.cut(sentence))
        tags = list(tagging(words))
        words = [w for w, _ in tags]
        tags = [tag for _, tag in tags]
        yield words, tags
        count += 1
