import re
from itertools import cycle
from conf import CORPUS_LIST
from regularize import regularize_punct
from utils import join_words2, strip


def generate_sentences():
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
            line = strip(line, '\n\t ')
            if line:
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
