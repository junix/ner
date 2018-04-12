from . import word2vec


class Word2Vec(word2vec.Word2Vec):
    _model = None

    def __init__(self):
        super(Word2Vec, self).__init__()

    def _ensure_model_loaded(self):
        import gensim
        from conf import GENSIM_MODEL_PATH
        if self._model is None:
            self._model = gensim.models.Word2Vec.load(GENSIM_MODEL_PATH)
        self._detect_none_word_vec()

    def __repr__(self):
        return "<gensim word2vec>"

    def get_raw_word2vec(self, word):
        return self._model.wv[word]
