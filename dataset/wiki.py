import re
from conf import WIKI_DATASET
from regularize.replace import regularize_punct


def generate_sentences():
    for line in open(WIKI_DATASET):
        yield from generate_sentence(line)


def generate_sentence(text):
    acc = []
    for c in text:
        if c in '，、。（）':
            sentence, acc = regularize_punct(''.join(acc)), []
            if len(sentence) > 4:
                yield sentence
        else:
            acc.append(c)
