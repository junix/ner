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
    "找到",
    "找得到",
    "找找",
    "搜",
    "搜一下",
    "搜到",
    "搜寻",
    "搜寻一下",
    "搜索",
    "搜索一下",
    "有",
    "有几个",
    "有多少",
    "有没有",
    "查",
    "查一下",
    "查找",
    "查找",
    "查询",
    "查询一下",
    "这里有",
    "这里有多少",
)
entity_patterns = (
    "<{keyword}>",
    "<{keyword}>",
    "<{keyword}>",
    "<{keyword}>",
    "<{keyword}>",
    "<{keyword}>演讲",
    "<{keyword}>的内容",
    "<{keyword}>的演讲",
    "<{keyword}>的知识",
    "<{keyword}>方面的",
    "<{keyword}>的讲座",
    "<{keyword}>的课程",
    "<{keyword}>相关",
    "<{keyword}>相关的",
    "<{keyword}>类",
    "<{keyword}>讲座",
    "<{keyword}>课",
    "<{keyword}>课程",
    "介绍<{keyword}>",
    "介绍<{keyword}>的",
    "关于<{keyword}>",
    "有关<{keyword}>",
    "有关<{keyword}>的",
    "讲述<{keyword}>",
    "讲述<{keyword}>的",
    "课程<{keyword}>",
)

hellos = (
    "",
    "hi",
    "hello",
    "不知道",
    "你",
    "亲",
    "你们",
    "你们这里",
    "你们云学堂",
    "这里",
    "哪里",
    "你好",
    "喂",
    "嗨",
    "哈啰",
    "我说",
    "敢问",
    "请",
    "请你",
    "请问",
    "烦请",
    "问一下",
    "麻烦",
    "麻烦一下",
    "麻烦你",
    "麻烦问一下",
    "你好小乐",
    "小乐",
)

seconds = (
    "",
    "",
    "可不可以",
    "可否",
    "帮我",
    "是否",
    "替我",
    "知不知道",
    "能不能",
    "能否",
    "怎么",
    "怎样",
    "有没有办法",
    "才能",
    "请",
    "请你",
    "这里",
    "什么地方能",
)

tailers = (
    "",
    "",
    "",
    "?",
    "可以吗",
    "可以吗",
    "吗",
    "好吗",
    "好吗",
    "行吗",
)

puncts = ('', ',', '，', '.', '。', '!', '！', '?', '？')


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

        if h:
            h += random.choice(puncts)
        if t:
            t += random.choice(puncts)
        sentence = h + s + op + entity.format(keyword=w) + t
        words = list(jieba.cut(sentence))
        tags = list(tagging(words))
        words = [w for w, _ in tags]
        tags = [tag for _, tag in tags]
        yield words, tags
        count += 1
