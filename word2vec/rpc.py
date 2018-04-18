from grpc_server.client import get_word2vec
from . import word2vec


class Word2Vec(word2vec.Word2Vec):
    _model = None

    def __init__(self):
        super(Word2Vec, self).__init__()

    def _ensure_model_loaded(self):
        self._detect_none_word_vec()

    def get_batch(self, items):
        self._ensure_model_loaded()
        return get_word2vec(items)

    def __repr__(self):
        return "<rpc word2vec>"

    def get_raw_word2vec(self, word):
        rep = get_word2vec([word])
        return rep[0]
