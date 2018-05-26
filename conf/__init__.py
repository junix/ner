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

MODEL_PERSIST_DIR = os.path.dirname(__file__) + '/../model_persist'
LANG_DUMP_FILE = MODEL_PERSIST_DIR + '/lang.pt'
MODEL_DUMP_FILE = MODEL_PERSIST_DIR + '/model.dump'

WIKI_DATASET = '/home/wanglijun/corpus/std_zh_wiki'


