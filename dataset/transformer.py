import numpy as np

from word2vec import wv


def transform(words):
    return np.concatenate([wv[w].reshape(1, -1) for w in words])
