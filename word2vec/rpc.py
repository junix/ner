import grpc
import numpy as np
from proto import data_pb2, data_pb2_grpc
from conf import WORD2VEC_RPC_SERVER_HOST, WORD2VEC_RPC_SERVER_PORT
from .word2vec import Word2Vec as BaseWord2Vec

_client = None


def _rep_to_numpy(response):
    return [np.array(vec.vec).astype(np.float32) for vec in response.vec_seq]


def make_request(words):
    return data_pb2.WordSeq(word_seq=words)


def get_word2vec(sentences):
    if _client is None:
        _make_client()
    for i in range(3):
        try:
            response = _client.DoGetWord2Vec(make_request(sentences))
            return _rep_to_numpy(response)
        except Exception as e:
            print(e)
            _make_client()


def _make_client():
    global _client
    chan = grpc.insecure_channel(WORD2VEC_RPC_SERVER_HOST + ':' + WORD2VEC_RPC_SERVER_PORT)
    _client = data_pb2_grpc.GetWord2VecStub(channel=chan)
    return _client


class Word2Vec(BaseWord2Vec):
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
