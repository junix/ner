from itertools import cycle

import random
from yxt_nlp_toolkit.utils import regularize_punct, join_words, all_punctuations

from conf import CORPUS_LIST


def generate_sentences(sample_ratio=0.5):
    assert .0 <= sample_ratio <= 1.
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


def remove_punctuation(text):
    for ch in text:
        if ch in all_punctuations:
            if ch in "，。？！;：…":
                yield '。'
        else:
            yield ch


def split_to_pieces(text):
    for sentence in regularize_punct(remove_punctuation(text)).split('。'):
        sentence = _rejoin(sentence).strip()
        if len(sentence) > 4:
            yield sentence
