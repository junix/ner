import torch
import os

"""
基本的配置
"""

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def use_cpu():
    global DEVICE
    DEVICE = torch.device('cpu')


def use_cuda():
    global DEVICE
    DEVICE = torch.device('cuda')


MODEL_ZOO = os.path.dirname(__file__) + '/../model_zoo'
LANG_PT_FILE = MODEL_ZOO + '/lang.pt'
MODEL_PT_FILE = MODEL_ZOO + '/model.dump'
DEFAULT_MODEL_PT_NAME = 'model.pt'

CORPUS_LIST = (
    '/home/wanglijun/corpus/std_zh_wiki',
    # '/home/wanglijun/corpus/corpus.txt',
)
