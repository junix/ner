import jieba
import random

from dataset.corpus import generate_sentences as gen_real_sentences
from dataset.fake_dataset import generate_a_faked_yxt_query, \
    generate_a_faked_query


def generate_dataset():
    for sentence, faked in generate_tagged_sentences():
        words = [w for w in jieba.cut(sentence) if w]
        tags = list(tagging(words))
        words = [w for w, _ in tags]
        tags = [tag for _, tag in tags]
        yield words, tags, faked


def generate_tagged_sentences():
    corpus_gen = generate_real_tagged_sentence()
    while True:
        rnd = random.randint(0, 100)
        if rnd < 5:
            yield generate_a_faked_yxt_query(), True
        elif rnd < 50:
            yield corpus_gen.send(None), False
        else:
            yield generate_a_faked_query(), True


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


def generate_real_tagged_sentence():
    while True:
        for sentence in gen_real_sentences():
            yield '<{keyword}>'.format(keyword=sentence)


def keyword_of(words, tags):
    keyword = ''.join([w for w, t in zip(words, tags) if t == 1])
    category = ''.join([w for w, t in zip(words, tags) if t == 2])
    return category, keyword


def output_seq_of(words, tags):
    return [w for w, t in zip(words, tags) if t != 0]
