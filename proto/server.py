#! /usr/bin/env python
# -*- coding: utf-8 -*-
import grpc
import time
from concurrent import futures
from proto import data_pb2, data_pb2_grpc
from word2vec import Word2Vec

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_HOST = 'localhost'
_PORT = '8080'

_word2vec = Word2Vec()


class GetWord2Vec(data_pb2_grpc.GetWord2VecServicer):

    def get_word2vec(self, word):
        return data_pb2.Word2Vec(vec=list(_word2vec[word]))

    def DoGetWord2Vec(self, request, context):
        vec_seq = []
        for word in request.word_seq:
            vec_seq.append(self.get_word2vec(word))
        return data_pb2.Word2VecSeq(vec_seq=vec_seq)


def serve():
    grpcServer = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    data_pb2_grpc.add_GetWord2VecServicer_to_server(GetWord2Vec(), grpcServer)
    grpcServer.add_insecure_port(_HOST + ':' + _PORT)
    grpcServer.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        grpcServer.stop(0)


if __name__ == '__main__':
    serve()
