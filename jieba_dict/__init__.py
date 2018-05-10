import os
import jieba

JIEBA_USER_DICT = os.path.dirname(__file__) + '/dict.dat'

_loaded_dict = False


def init_user_dict():
    global _loaded_dict
    if _loaded_dict:
        return
    jieba.load_userdict(JIEBA_USER_DICT)
    _loaded_dict = True
