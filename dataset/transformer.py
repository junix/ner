import numpy as np
# import time

from word2vec import wv


def transform(words):
    # beg = time.time()
    vecs = wv.get_batch(words)
    # print(len(words), '->', time.time() - beg)
    return np.array(vecs)
