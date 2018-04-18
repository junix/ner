import numpy as np

from word2vec import wv


def transform(words):
    vecs = wv.get_batch(words)
    return np.array(vecs)
