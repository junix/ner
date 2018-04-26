import os
import jieba

JIEBA_USER_DICT = os.path.dirname(__file__) + '/dict.dat'
jieba.load_userdict(JIEBA_USER_DICT)


def init_user_dict():
    pass
