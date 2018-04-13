import os
import re
import jieba
import random

_current_dir = os.path.dirname(__file__)
_new_hans_dict = _current_dir + '/' + 'chinese_words.txt'

search_ops = (
    "",
    "找",
    "找找",
    "找一下",
    "找到",
    "找得到",
    "搜",
    "搜搜",
    "搜一下",
    "搜寻",
    "搜寻一下",
    "搜索",
    "搜索一下",
    "有",
    "有几个",
    "有多少",
    "有没有",
    "查",
    "查查",
    "查一下",
    "查找",
    "查询",
    "查询一下",
    "这里有",
    "哪里有",
    "这里有多少",
)
simple_entity_patterns = (
    "<{keyword}>",
)

composed_entity_patterns = (
    "<{keyword}>演讲",
    "<{keyword}>的内容",
    "<{keyword}>的演讲",
    "<{keyword}>的知识",
    "<{keyword}>方面的",
    "<{keyword}>的讲座",
    "<{keyword}>的课程",
    "<{keyword}>的文档",
    "<{keyword}>的资料",
    "<{keyword}>的视频",
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

entity_patterns = simple_entity_patterns + composed_entity_patterns

hellos = (
    "",
    "hi",
    "hello",
    "不知道",
    "你",
    "亲",
    "你们",
    "你们这里",
    "这里",
    "哪里",
    "你好",
    "喂",
    "喔",
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


def _read_words():
    with open(_new_hans_dict, 'r') as f:
        for line in f:
            try:
                # word, tag, *_ = line.split(' \t')
                word, *_ = re.split('\s', line)
                yield word
                # if tag and tag[0] in ('n', 'i', 'v', 'g', 'l', 'd'):
                #     yield word
            except Exception as e:
                print(e)
                pass


_all_words = tuple(set(_read_words()))
_all_words_and_puncts = _all_words + puncts


def fake_sentence():
    word_len = random.randint(0, 4)
    return ''.join([random.choice(_all_words_and_puncts) for _ in range(word_len)])


def tagging(words):
    next_mark = 0
    for w in words:
        if w == '<':
            next_mark = 1
        elif w == '>':
            next_mark = 0
        else:
            yield w, next_mark


def generate_sentence():
    while True:
        ws = _all_words
        w = random.choice(ws)
        h = random.choice(hellos)
        s = random.choice(seconds)
        op = random.choice(search_ops)

        if op:
            entity = random.choice(entity_patterns)
        else:
            entity = random.choice(composed_entity_patterns)

        t = random.choice(tailers)

        if h:
            h += random.choice(puncts)
        if t:
            t += random.choice(puncts)

        noise = ""
        if (h or s) or search_ops and random.randint(0, 20) == 0:
            noise = fake_sentence()
            if random.randint(0, 5) == 0:
                noise += random.choice(puncts)

        yield noise + h + s + op + entity.format(keyword=w) + t


def generate_dataset():
    for sentence in generate_sentence():
        words = list(jieba.cut(sentence))
        tags = list(tagging(words))
        words = [w for w, _ in tags]
        tags = [tag for _, tag in tags]
        yield words, tags
