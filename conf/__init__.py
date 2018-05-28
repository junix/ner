import torch
import os

"""
基本的配置
"""

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

GENSIM_MODEL_PATH = "/data/word2vec_models/200/model.bin"

word2vec_rpc_servers = (
    ('localhost', 9191),
    ('localhost', 9190),
)

MODEL_ZOO = os.path.dirname(__file__) + '/../model_zoo'
LANG_PT_FILE = MODEL_ZOO + '/lang.pt'
MODEL_PT_FILE = MODEL_ZOO + '/model.dump'
DEFAULT_MODEL_PT_NAME = 'model.pt'

CORPUS_LIST = (
    '/home/wanglijun/corpus/std_zh_wiki',
    # '/home/wanglijun/corpus/corpus.txt',
)


