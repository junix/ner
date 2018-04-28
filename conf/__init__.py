import torch

"""
基本的配置
"""

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

GENSIM_MODEL_PATH = "/opt/word2vec_models/200/model.bin"

word2vec_rpc_servers = (
    ('localhost', 9191),
    ('localhost', 9190),
)
