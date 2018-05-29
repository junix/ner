import os
from .context import *

"""
基本的配置
"""

MODEL_ZOO = os.path.dirname(__file__) + '/../model_zoo'
LANG_PT_FILE = MODEL_ZOO + '/lang.pt'
MODEL_PT_FILE = MODEL_ZOO + '/model.dump'
DEFAULT_MODEL_PT_NAME = 'model.pt'

CORPUS_LIST = (
    '/home/wanglijun/corpus/std_zh_wiki',
    # '/home/wanglijun/corpus/corpus.txt',
)
