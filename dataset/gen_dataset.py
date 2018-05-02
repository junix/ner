import os
import re
import numpy as np
import jieba_dict
import jieba
import random
from utils.str_algo import regularize_punct
from .transformer import transform

_current_dir = os.path.dirname(__file__)
_new_hans_dict = _current_dir + '/' + 'chinese_words.txt'

jieba_dict.init_user_dict()

search_ops = (
    '{please}找{adv}',
    '{please}找找',
    '找到',
    '找得到',
    '{please}找一找',
    '{please}搜{adv}',
    '{please}搜搜',
    '{please}搜一搜',
    '{please}搜寻{adv}',
    '{please}搜索{adv}',
    '有',
    '有没有',
    '有几个',
    '有多少',
    '有没有',
    '有那些',
    '{please}查{adv}',
    '{please}查一查',
    '{please}查查',
    '{please}查找{adv}',
    '{please}查询{adv}',
    '{please}询找{adv}',
    '这里有',
    '哪里有',
    '{please}看{adv}',
    '{please}听{adv}',
    '{please}打开',
    '{please}跳到',
    '{please}回到',
    '{please}登录到',
)

pleases = (
    '我要',
    '我想',
    '帮我', '替我', '给我',
    '想',
    '请',
    '请你',
    '麻烦你',
    '能',
    '能不能',
    '可否',
    '可以',
    '可不可以',
)

search_op_advs = (
    "一下",
)


def get_a_search_op():
    if random.randint(0, 10) <= 4:
        return ''
    op = random.choice(search_ops)
    if op:
        adv, please = '', ''
        if random.randint(0, 10) < 2:
            adv = random.choice(search_op_advs)
        if random.randint(0, 10) < 1:
            please = random.choice(pleases)
        op = op.format(adv=adv, please=please)
    return op


simple_entity_patterns = (
    "<{keyword}>",
)

entity_field_prefixes = (
    '[文档]',
    '[文章]',
    '[演讲]',
    '[讲话]',
    '[知识]',
    '[视频]',
    '[讲座]',
    '[课]',
    '[课程]',
    '[教程]',
    '[资料]',
    '[材料]',
    '[电影]',
    '[书籍]',
    '[商品]',
    '[素材]',
)

entity_field_suffixes = (
                            '内容',
                            '方面',
                            '相关',
                            '类',
                        ) + entity_field_prefixes

complex_entity_fields = {
    '方面',
    '相关',
    '类',
}

des = ('的', '', '')


def get_a_entity_field():
    field = random.choice(entity_field_suffixes)
    if field in complex_entity_fields and random.randint(0, 1) == 0:
        de = random.choice(des)
        concrete_field = None
        while concrete_field is None:
            concrete_field = random.choice(entity_field_suffixes)
            if concrete_field in complex_entity_fields:
                concrete_field = None
        field = field + de + concrete_field
    if '的' in field:
        return field
    return random.choice(des) + field


def get_a_composed_entity_pattern():
    v = random.randint(0, 100)
    if v <= 33:
        return get_a_about() + '<{keyword}>'
    if v <= 60:
        return random.choice(entity_field_prefixes) + '<{keyword}>'
    if v <= 85:
        return '<{keyword}>' + get_a_entity_field()
    return get_a_about() + '<{keyword}>' + get_a_entity_field()


abouts = (
    '关于',
    '有关',
    '有关于',
)

intros = (
    '介绍',
    '讲述',
    '叙述',
)


def get_a_about():
    rnd = random.randint(0, 100)
    if rnd < 33:
        return random.choice(abouts)
    if rnd < 66:
        return random.choice(intros)
    else:
        return random.choice(intros) + random.choice(abouts)


def get_a_entity_pattern():
    rnd = random.randint(0, 100)
    if rnd <= 20:
        return random.choice(simple_entity_patterns)
    return get_a_composed_entity_pattern()


tails = (
    '?',
    '可以吗',
    '怎么样',
    '可否',
    '吗',
    '好吗',
    '好不',
    '行吗',
    '行不',
)


def get_a_tail():
    rnd = random.randint(0, 100)
    if rnd <= 80:
        return ''
    return random.choice(tails)


puncts = ('', ',', '.', '!', '?')


def _read_words():
    with open(jieba_dict.JIEBA_USER_DICT, 'r') as f:
        for line in f:
            try:
                word, *_ = re.split('\s', line)
                yield word
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
        elif w == '[':
            next_mark = 2
        elif w in ('>', ']'):
            next_mark = 0
        else:
            yield w, next_mark


hellos = (
    '',
    'hi',
    'hello',
    '不知道',
    '你',
    '亲',
    '你们',
    '你们这里',
    '这里',
    '哪里',
    '你好',
    '喂',
    '喔',
    '嗨',
    '哈啰',
    '我说',
    '敢问',
    '请',
    '请你',
    '请问',
    '烦请',
    '问一下',
    '麻烦',
    '麻烦一下',
    '麻烦你',
    '麻烦问一下',
    '你好小乐',
    '小乐',
)


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
        op = get_a_search_op()

        if op:
            entity = get_a_entity_pattern()
        else:
            entity = get_a_composed_entity_pattern()

        t = get_a_tail()

        noise = ""
        if search_ops and random.randint(0, 10) <= 4:
            noise = fake_sentence(max_word_cnt=2)
            if random.randint(0, 5) <= 1:
                noise += random.choice(puncts)

        sentence = noise + h + op + entity.format(keyword=w) + t
        sentence = regularize_punct(sentence)
        yield sentence


def generate_dataset():
    for sentence in generate_sentence():
        words = list(jieba.cut(sentence))
        tags = list(tagging(words))
        words = [w for w, _ in tags]
        tags = [tag for _, tag in tags]
        yield words, tags




def load_dataset():
    from word2vec.gensims import Word2Vec
    embed = Word2Vec()
    dataset = generate_dataset()
    for xs, y_true in dataset:
        yield transform(xs, embed=embed), np.array(y_true)


def keyword_of(words, tags):
    keyword = ''.join([w for w, t in zip(words, tags) if t == 1])
    category = ''.join([w for w, t in zip(words, tags) if t == 2])
    return category, keyword


def output_seq_of(words, tags):
    return [w for w, t in zip(words, tags) if t != 0]
