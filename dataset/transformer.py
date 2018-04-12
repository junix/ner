import pickle
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

from word2vec import Word2Vec, wv
import numpy as np

__all__ = [
    'WordEmbeddingTransformer',
    'PosTagTransformer',
    'TargetTransformer',
    'PhraseCut',
    'PhraseTransformer',
]


class WordEmbeddingTransformer(TransformerMixin):
    def __init__(self, word2vec=None):
        self.word2vec = Word2Vec() if word2vec is None else word2vec

    def fit(self, _):
        return self

    def transform(self, xs):
        return np.array([self.word2vec[w] for w in xs])

    def __repr__(self):
        return '<WordEmbeddingTransformer>'


class PosTagTransformer(TransformerMixin):
    def __init__(self):
        self.encoder = DictVectorizer(sparse=False)

    def fit(self, xs):
        self.encoder.fit(self._to_dict_fmt(xs))
        return self

    def transform(self, xs):
        return np.array(self.encoder.transform(self._to_dict_fmt(xs))).astype(np.float32)

    @staticmethod
    def _to_dict_fmt(xs):
        return [{'pos_tag': e} for e in xs]


class TargetTransformer(TransformerMixin):
    def __init__(self):
        self.encoder = LabelEncoder()
        self.encoder.fit(('Y', 'N'))

    def fit(self, x):
        self.encoder.fit(list(set(x)))
        return self

    def transform(self, x):
        return np.array(self.encoder.transform(x), dtype=np.float32)


class PhraseCut(TransformerMixin):
    def fit(self, _xs):
        return self

    @staticmethod
    def _make_frame(text):
        from jieba.posseg import cut
        if len(text) == 2:
            first, *remain = list(cut(text))
            if remain:
                return pd.DataFrame([{'word': text, 'pos_tag': 'nil'}])
            else:
                return pd.DataFrame([{'word': first.word, 'pos_tag': first.flag}])
        return pd.DataFrame([{'word': e.word, 'pos_tag': e.flag} for e in cut(text)])

    def transform(self, xs):
        from multiprocessing import Pool
        with Pool(20) as p:
            return p.map(PhraseCut._make_frame, xs)


class PhraseTransformer(TransformerMixin):
    def __init__(self, load_file=None):
        self.cutter = PhraseCut()
        self.target_encoder = TargetTransformer()
        self.word_embedding = WordEmbeddingTransformer()
        if load_file is None:
            self.postag_encoder = PosTagTransformer()
        else:
            self._load(load_file)

    def fit(self, xs, _ys=None):
        postags = pd.concat([cs['pos_tag'] for cs in self.cutter.fit_transform(xs)])
        self.postag_encoder.fit(postags)
        return self

    def transform(self, xs, ys=None):
        xss = []
        for cs in self.cutter.transform(xs):
            xss.append(self._transform_phrase(cs))

        # xs = np.array(xss, dtype=np.object)
        xs = xss
        if ys is None:
            return xs
        return xs, ys

    def dump(self, file_name):
        with open(file_name, "wb") as f:
            pickle.dump(self.postag_encoder, f)

    def _load(self, file_name):
        with open(file_name, "rb") as f:
            self.postag_encoder = pickle.load(f)

    def _transform_phrase(self, ps):
        word_embedding = self.word_embedding.transform(ps['word'])
        postags = self.postag_encoder.transform(ps['pos_tag'])
        return np.concatenate([word_embedding, postags], axis=1)

    def output_feature_len(self):
        xs = ['中国', '技术']
        for elem in self.transform(xs):
            return elem.shape[1]


def transform(words):
    return np.concatenate([wv[w].reshape(1, -1) for w in words])
