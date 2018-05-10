import numpy as np


class Word2Vec:
    def __init__(self):
        self._none_word_vec = None

    def _detect_none_word_vec(self):
        if self._none_word_vec is not None:
            return
        for word in ('中国', '知识', '能力'):
            try:
                self._none_word_vec = np.zeros(len(self.get_raw_word2vec(word)))
                return
            except KeyError:
                pass
        raise ValueError('Fail to detect wordvec len')

    def __getitem__(self, item):
        self._ensure_model_loaded()
        try:
            vec = self.get_raw_word2vec(item)
            return np.array(vec)
        except KeyError:
            return self._none_word_vec.copy()

    def get_batch(self, items):
        raise NotImplementedError

    def _ensure_model_loaded(self):
        raise NotImplementedError

    def get_raw_word2vec(self, word):
        raise NotImplementedError
