import grpc
import random
import numpy as np
from error import NerError
from proto import data_pb2, data_pb2_grpc
from conf import word2vec_rpc_servers
from .word2vec import Word2Vec as BaseWord2Vec
from log import log

_cached_client = None


def _rep_to_numpy(response):
    return [np.array(vec.vec).astype(np.float32) for vec in response.vec_seq]


def _make_request(words):
    return data_pb2.WordSeq(word_seq=words)


def get_word2vec(words):
    global _cached_client

    req = _make_request(words)
    for c in _client_seq():
        try:
            reply = c.DoGetWord2Vec(req)
            result = _rep_to_numpy(reply)
            _cached_client = c
            return result
        except grpc.RpcError:
            log.exception('rpc call error')
    raise NerError('server unavailable')


def _client_seq():
    if _cached_client is not None:
        yield _cached_client
    servers = list(word2vec_rpc_servers)
    random.shuffle(servers)
    for host, port in servers:
        address = '{host}:{port}'.format(host=host, port=port)
        chan = grpc.insecure_channel(address)
        stub = data_pb2_grpc.GetWord2VecStub(channel=chan)
        yield stub


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
