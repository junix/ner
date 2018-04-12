import os
import jieba
import random

_current_dir = os.path.dirname(__file__)
_new_hans_dict = _current_dir + '/' + 'new_hans.txt'

search_ops = (
    "",
    "",
    "",
    "找一下",
    "查",
    "查一下",
    "搜",
    "搜一下",
    "查询",
    "查询一下",
    "搜索",
    "搜索一下",
    "搜寻",
    "搜寻一下",
    "搜索",
    "查找",
    "找得到",
    "查找",
    "搜索一下",
    "有",
    "有没有",
    "有多少",
    "有几个",
    "这里有多少",
    "这里有",
    "找找",
    "找到",
    "搜到",
)
entity_patterns = (
    "<{keyword}>",
    "<{keyword}>",
    "<{keyword}>",
    "<{keyword}>",
    "<{keyword}>",
    "课程<{keyword}>",
    "关于<{keyword}>",
    "有关<{keyword}>的",
    "介绍<{keyword}>的",
    "讲述<{keyword}>的",
    "有关<{keyword}>",
    "介绍<{keyword}>",
    "讲述<{keyword}>",
    "<{keyword}>相关",
    "<{keyword}>相关的",
    "<{keyword}>类",
    "<{keyword}>课程",
    "<{keyword}>的课程",
    "<{keyword}>讲座",
    "<{keyword}>的讲座",
)

hellos = (
    "请问",
    "请",
    "请你",
    "麻烦问一下",
    "敢问",
    "麻烦",
    "麻烦一下",
    "你好",
    "Hi",
    "嗨",
    "喂",
    "问一下",
    "你",
    "你们这里",
    "麻烦你",
    "",
)

seconds = (
    "",
    "",
    "请",
    "请你",
    "能不能",
    "可不可以",
    "知不知道",
    "可否",
    "能否",
    "是否",
    "帮我",
    "替我",
    "这里",
)

tailers = (
    "",
    "",
    "",
    "好吗",
    "可以吗",
    "好吗",
    "可以吗",
    "行吗",
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


def generate_dataset(size=20000000):
    ws = list(read_keywords())
    count = 0
    while count < size:
        w = random.choice(ws)
        h = random.choice(hellos)
        s = random.choice(seconds)
        op = random.choice(search_ops)
        entity = random.choice(entity_patterns)
        t = random.choice(tailers)
        if random.randint(0, 1) == 0 and h:
            h = h + ','
        if random.randint(0, 1) == 0 and t:
            t = t + ','
        sentence = h + s + op + entity.format(keyword=w) + t
        words = list(jieba.cut(sentence))
        tags = list(tagging(words))
        words = [w for w, _ in tags]
        tags = [tag for _, tag in tags]
        yield words, tags
        count += 1
