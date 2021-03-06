import pickle
import conf


class Lang:
    NIL, NIL_INDEX = '<nil>', 0

    def __init__(self, lang_name, words):
        self.lang_name = lang_name
        self.word2index = {Lang.NIL: Lang.NIL_INDEX}
        for index, w in enumerate(set(words), 1):
            self.word2index[w] = index

    def __repr__(self):
        return '<{lang}>'.format(lang=self.lang_name)

    def __getitem__(self, item):
        return self.to_index(item)

    def vocab_size(self):
        return len(self.word2index)

    def to_index(self, words):
        if isinstance(words, (list, tuple)):
            return tuple(self.word2index.get(w, Lang.NIL_INDEX) for w in words)
        elif isinstance(words, str):
            return self.word2index.get(words, Lang.NIL_INDEX)
        raise TypeError('idx only support list,tuple,str,but found:{}'.format(words))

    def dump(self, name='lang.pt'):
        with open(conf.path_in_zoo(name), 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, name='lang.pt'):
        with open(conf.path_in_zoo(name), 'rb') as f:
            return pickle.load(f)
