import os
from .context import *

"""
基本的配置
"""


def path_in_zoo(name):
    model_zoo = os.path.dirname(__file__) + '/../model_zoo'
    return '{}/{}'.format(model_zoo, name)


CORPUS_LIST = (
    '/home/wanglijun/corpus/std_zh_wiki',
    # '/home/wanglijun/corpus/corpus.txt',
)
