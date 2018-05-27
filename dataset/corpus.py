import re
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
        yield from split_to_pieces(line)


def _rejoin(sentence):
    if not sentence:
        return sentence
    words = sentence.split(' ')
    return join_words2(words)


def split_to_pieces(text):
    for sentence in re.split(r'[,。､.;:?!#()\n\t\"]|……', regularize_punct(text)):
        sentence = _rejoin(sentence).strip()
        if len(sentence) > 4:
            yield sentence