import os
import jieba
from itertools import cycle

JIEBA_USER_DICT = os.path.dirname(__file__) + '/dict.dat'
STOPWORDS_LIST = os.path.dirname(__file__) + '/stopwords.dat'

_loaded_dict = False


def init_user_dict():
    global _loaded_dict
    if _loaded_dict:
        return
    jieba.load_userdict(JIEBA_USER_DICT)
    _loaded_dict = True


def _read_stopwords():
    with open(STOPWORDS_LIST, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


_stopword_set = set(_read_stopwords())


def is_stopword(word):
    return word in _stopword_set
