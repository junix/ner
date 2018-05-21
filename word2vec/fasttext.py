import fasttext
from . import word2vec


class Word2Vec(word2vec.Word2Vec):
    _MODEL_DATA_FILE = '/Users/junix/ml/wiki/wiki.zh.bin'

    def __init__(self):
        self._model = None
        super(Word2Vec, self).__init__()

    def _ensure_model_loaded(self):
        if self._model is None:
            self.model = fasttext.load_model(self._MODEL_DATA_FILE)
            self._detect_none_word_vec()

    def get_raw_word2vec(self, word):
        self._ensure_model_loaded()
        return self._model[word]
