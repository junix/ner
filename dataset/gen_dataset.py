import os
import re
import jieba
import random

_current_dir = os.path.dirname(__file__)
_new_hans_dict = _current_dir + '/' + 'chinese_words.txt'

search_ops = (
    "",
    "{who}找{adv}",
    "{who}找找",
    "找到",
    "找得到",
    "{who}搜{adv}",
    "{who}搜搜",
    "{who}搜寻{adv}",
    "{who}搜索{adv}",
    "有",
    "有几个",
    "有多少",
    "有没有",
    "{who}查{adv}",
    "{who}查查",
    "{who}查找{adv}",
    "{who}查询{adv}",
    "这里有",
    "哪里有",
    "{who}看{adv}",
    "{who}找{adv}",
    "{who}听{adv}",
    "这里有多少",
)

whos = (
    '我要',
    '我想',
    '想',
    '能',
    '能不能',
    '可否',
    '可以',
)

search_op_advs = (
    "一下",
)


def get_a_search_op():
    op = random.choice(search_ops)
    if op:
        adv, who = '', ''
        if random.randint(0, 10) < 2:
            adv = random.choice(search_op_advs)
        if random.randint(0, 10) < 3:
            who = random.choice(whos)
        op = op.format(adv=adv, who=who)
    return op


simple_entity_patterns = (
    "<{keyword}>",
)

entity_fields = (
    "内容",
    "文档",
    "文章",
    "方面",
    "演讲",
    "讲话",
    "相关",
    "知识",
    "类",
    "视频",
    "讲座",
    "课",
    "课程",
    "资料",
    "材料",
    "电影",
    "书籍",
    "商品",
    "素材",
)

complex_entity_fields = {
    "方面",
    "相关",
    "类",
}

des = ('的', '', '')


def get_a_entity_field():
    field = random.choice(entity_fields)
    if field in complex_entity_fields and random.randint(0, 1) == 0:
        de = random.choice(des)
        concrete_field = None
        while concrete_field is None:
            concrete_field = random.choice(entity_fields)
            if concrete_field in complex_entity_fields:
                concrete_field = None
        field = field + de + concrete_field
    if '的' in field:
        return field
    return random.choice(des) + field


def get_a_compose_segment(segments):
    acc = ""
    while True:
        if not acc:
            acc = random.choice(segments)
        if '{self}' in acc:
            acc = acc.format(self=random.choice(segments))
        else:
            return acc


def get_a_composed_entity_pattern():
    v = random.randint(0, 10)
    if v <= 2:
        return get_a_about() + '<{keyword}>'
    if v <= 8:
        return "<{keyword}>" + get_a_entity_field()

    return get_a_about() + '<{keyword}>' + get_a_entity_field()


abouts = (
    "关于",
    "有关",
    "有关于",
    "介绍",
    "介绍{self}",
    "讲述",
    "讲述{self}",
    "叙述",
    "叙述{self}",
    "课程",
)


def get_a_about():
    return get_a_compose_segment(abouts)


def get_a_entity_pattern():
    if random.randint(0, 3) <= 0:
        return random.choice(simple_entity_patterns)
    return get_a_composed_entity_pattern()


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
    # "帮我",
    "是否",
    # "替我",
    "我",
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
    "怎么样",
    "可否",
    "吗",
    "好吗",
    "好不",
    "行吗",
    "行不",
)

puncts = ('', ',', '，', '.', '。', '!', '！', '?', '？')


def _read_words():
    with open(_new_hans_dict, 'r') as f:
        for line in f:
            try:
                # word, ner, *_ = line.split(' \t')
                word, *_ = re.split('\s', line)
                yield word
                # if ner and ner[0] in ('n', 'i', 'v', 'g', 'l', 'd'):
                #     yield word
            except Exception as e:
                print(e)
                pass


_all_words = tuple(set(_read_words()))
_all_words_and_puncts = _all_words + puncts


def fake_sentence(min_word_cnt=0, max_word_cnt=4, with_punct=True):
    word_len = random.randint(min_word_cnt, max_word_cnt)
    if with_punct:
        return ''.join([random.choice(_all_words_and_puncts) for _ in range(word_len)])
    else:
        return ''.join([random.choice(_all_words) for _ in range(word_len)])


def tagging(words):
    next_mark = 0
    for w in words:
        if w == '<':
            next_mark = 1
        elif w == '>':
            next_mark = 0
        else:
            yield w, next_mark


def make_hello():
    if random.randint(0, 10) <= 9:
        return ""
    h = random.choice(hellos)
    if h and random.randint(0, 10) <= 5:
        h += random.choice(puncts)
    return h


def generate_sentence():
    while True:
        ws = _all_words
        w = fake_sentence(min_word_cnt=1, max_word_cnt=3, with_punct=False)
        h = make_hello()
        # s = random.choice(seconds)
        op = get_a_search_op()

        if op:
            entity = get_a_entity_pattern()
        else:
            entity = get_a_composed_entity_pattern()

        t = random.choice(tailers)

        # if h:
        #     h += random.choice(puncts)
        # if t:
        #     t += random.choice(puncts)

        noise = ""
        # if (h or s) or search_ops and random.randint(0, 20) == 0:
        if search_ops and random.randint(0, 10) <= 4:
            noise = fake_sentence(max_word_cnt=2)
            if random.randint(0, 5) <= 1:
                noise += random.choice(puncts)

        # yield noise + h + s + op + entity.format(keyword=w) + t
        yield noise + h + op + entity.format(keyword=w) + t


def generate_dataset():
    for sentence in generate_sentence():
        words = list(jieba.cut(sentence))
        tags = list(tagging(words))
        words = [w for w, _ in tags]
        tags = [tag for _, tag in tags]
        yield words, tags


def keyword_of(words, tags):
    return ''.join([w for w, t in zip(words, tags) if t == 1])
