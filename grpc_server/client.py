#! /usr/bin/env python
# -*- coding: utf-8 -*-
import grpc
import numpy as np
from grpc_server import data_pb2, data_pb2_grpc
from conf import WORD2VEC_RPC_SERVER_HOST, WORD2VEC_RPC_SERVER_PORT

_client = None


def to_numpy(response):
    return [np.array(vec.vec).astype(np.float32) for vec in response.vec_seq]


def make_request(words):
    return data_pb2.WordSeq(word_seq=words)


def get_word2vec(sentences):
    if _client is None:
        conn()
    for i in range(3):
        try:
            response = _client.DoGetWord2Vec(make_request(sentences))
            return to_numpy(response)
        except Exception as e:
            print(e)
            conn()


def conn():
    global _client
    chan = grpc.insecure_channel(WORD2VEC_RPC_SERVER_HOST + ':' + WORD2VEC_RPC_SERVER_PORT)
    _client = data_pb2_grpc.GetWord2VecStub(channel=chan)
    return _client


if __name__ == '__main__':
    req = ['你好', '请问']
    print(get_word2vec(req))
