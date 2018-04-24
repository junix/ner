import os
import jieba

_jieba_user_dict = os.path.dirname(__file__) + '/dict.dat'
jieba.load_userdict(_jieba_user_dict)
