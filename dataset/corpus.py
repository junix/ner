import re
import random

from yxt_nlp.utils import regularize_punct, join_words
from itertools import cycle
from conf import CORPUS_LIST


def generate_sentences(sample_ratio=0.5):
    assert .0 < sample_ratio < 1.
    while True:
        files = [open(corpus, 'r') for corpus in CORPUS_LIST]
        files_count = len(files)
        assert len(files) > 0, "no "
        cont_empty_line_count = 0
        for file in cycle(files):
            line = file.readline()
            if not line:
                cont_empty_line_count += 1
                if cont_empty_line_count > files_count * 2:
                    break
                continue
            cont_empty_line_count = 0
            line = line.strip('\n\t ')
            if not line:
                continue
            for piece in split_to_pieces(line):
                if random.random() <= sample_ratio:
                    yield piece


def _rejoin(sentence):
    if not sentence:
        return sentence
    words = sentence.split(' ')
    return join_words(words)


def split_to_pieces(text):
    for sentence in re.split(r'[,。､.;:?!#()\n\t\"]|……', regularize_punct(text)):
        sentence = _rejoin(sentence).strip()
        if len(sentence) > 4:
            yield sentence
