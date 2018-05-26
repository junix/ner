from itertools import cycle
from conf import CORPUS_LIST
from regularize import regularize_punct
from utils import join_words2, strip


def generate_sentences():
    files = [open(corpus, 'r') for corpus in CORPUS_LIST]
    assert len(files) > 0, "no "
    for file in cycle(files):
        line = file.readline()
        line = strip(line, '\n\t ')
        if not line:
            continue
        yield from generate_sentence(line)


def _rejoin(sentence):
    if not sentence:
        return sentence
    words = sentence.split(' ')
    return join_words2(words)


def generate_sentence(text):
    acc = []
    for c in text:
        if c in '，、!！。（）':
            sentence, acc = regularize_punct(''.join(acc)), []
            if len(sentence) > 4:
                yield strip(_rejoin(sentence), ' \n\t')
        else:
            acc.append(c)
