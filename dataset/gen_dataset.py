import jieba
import random

from dataset.corpus import generate_sentences as gen_real_sentences
from dataset.fake_dataset import generate_a_faked_yxt_query, \
    generate_a_faked_query


def generate_dataset(drop_n=0, real_corpus_sample=0.3):
    for sentence, faked in generate_tagged_sentences(real_corpus_sample):
        words = [w for w in jieba.cut(sentence) if w]
        tags = list(tagging(words))
        words = [w for w, _ in tags]
        tags = [tag for _, tag in tags]
        if not words:
            continue
        if drop_n > 0:
            drop_n -= 1
            continue

        yield words, tags, faked


def generate_tagged_sentences(real_corpus_sample):
    corpus_gen = generate_real_tagged_sentence(real_corpus_sample)
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


def generate_real_tagged_sentence(sample_ratio):
    for sentence in gen_real_sentences(sample_ratio):
        yield '<{keyword}>'.format(keyword=sentence)


def keyword_of(words, tags):
    keyword = ''.join([w for w, t in zip(words, tags) if t == 1])
    category = ''.join([w for w, t in zip(words, tags) if t == 2])
    return category, keyword


def output_seq_of(words, tags):
    return [w for w, t in zip(words, tags) if t != 0]
