from conf import WIKI_DATASET
from regularize import regularize_punct
from utils import join_words2


def generate_sentences():
    for line in open(WIKI_DATASET):
        yield from generate_sentence(line)


def _rejoin(sentence):
    if not sentence:
        return sentence
    words = sentence.split(' ')
    return join_words2(words)


def generate_sentence(text):
    acc = []
    for c in text:
        if c in '，、。（）':
            sentence, acc = regularize_punct(''.join(acc)), []
            if len(sentence) > 4:
                yield _rejoin(sentence)
        else:
            acc.append(c)
