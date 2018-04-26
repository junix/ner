import numpy as np
from word2vec import wv


def transform(words, embed=None):
    if embed:
        vecs = embed.get_batch(words)
    else:
        vecs = wv.get_batch(words)
    return np.array(vecs)
