import numpy as np


def transform(words, embed=None):
    from word2vec import wv
    if embed:
        vecs = embed.get_batch(words)
    else:
        vecs = wv.get_batch(words)
    return np.array(vecs)
