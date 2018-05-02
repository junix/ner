import os
import torch
import jieba

from conf import DEVICE
from dataset.gen_dataset import generate_sentence, tagging, output_seq_of
from dataset.transformer import transform

MAX_LENGTH = 20

SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def indexes_from_sentence(lang, words):
    return [lang.word2index[word] for word in words]


def tensor_from_sentence(lang, words):
    indexes = indexes_from_sentence(lang, words)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=DEVICE).view(-1, 1)


def build_lang_from_dict(file):
    lang = Lang()
    with open(file) as f:
        for line in f:
            lang.add_word(line.strip())
    return lang


lang = build_lang_from_dict(os.path.dirname(__file__) + '/../jieba_dict/dict.dat')


def generate_seq2seq_dataset():
    for sentence in generate_sentence():
        words = list(jieba.cut(sentence))
        tags = list(tagging(words))
        words = [w for w, _ in tags] + ['EOS']
        tags = [tag for _, tag in tags]
        yield torch.tensor(transform(words), dtype=torch.float32, device=DEVICE), tensor_from_sentence(lang,
                                                                                                       output_seq_of(
                                                                                                           words, tags))
