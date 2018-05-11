import re
import os

JIEBA_USER_DICT = os.path.dirname(__file__) + '/dict.dat'


def clean_dict():
    d = {}
    count = 0
    with open(JIEBA_USER_DICT, 'r') as f:
        for line in f:
            word, *left = re.split('\s', line)
            left = [e for e in left if e]
            attr = d.get(word, ())
            if len(left) < len(attr):
                continue
            d[word] = left
            count += 1

        print(count)
    with open(JIEBA_USER_DICT + '.new', 'w') as f:
        for k, v in d.items():
            fmt = ' '.join([k] + v)
            f.write(fmt + '\n')

clean_dict()
