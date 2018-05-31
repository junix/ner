import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import conf


class EntityRecognizer(nn.Module):

    def __init__(self, lang, embedding_dim=200, hidden_size=512, rnn_type='lstm', num_layers=2, ):
        super(EntityRecognizer, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = embedding_dim
        self.num_layers = num_layers
        self.lang = lang
        self.embedding = nn.Embedding(num_embeddings=lang.vocab_size, embedding_dim=embedding_dim)
        self.bidirectional = True
        self.rnn_type = rnn_type
        assert rnn_type in ('lstm', 'gru'), 'un-support rnn type:{}'.format(rnn_type)
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=embedding_dim,
                               hidden_size=hidden_size,
                               dropout=0.5,
                               num_layers=self.num_layers,
                               bidirectional=self.bidirectional)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=embedding_dim,
                              hidden_size=hidden_size,
                              dropout=0.5,
                              num_layers=self.num_layers,
                              bidirectional=self.bidirectional)

        self.hidden2tag = nn.Linear(hidden_size * 2 if self.bidirectional else 1, out_features=3)
        self.hidden = self.init_hidden()
        self.move_to_device(conf.device())

    def move_to_device(self, device):
        if device.type == 'cpu':
            self.cpu()
        else:
            self.cuda()

    def init_hidden(self):
        bidirect = 2 if self.bidirectional else 1
        if self.rnn_type == 'lstm':
            return (
                torch.zeros(self.num_layers * bidirect, 1, self.hidden_size, device=conf.device()),
                torch.zeros(self.num_layers * bidirect, 1, self.hidden_size, device=conf.device())
            )
        elif self.rnn_type == 'gru':
            return torch.zeros(self.num_layers * bidirect, 1, self.hidden_size, device=conf.device())

    def __getitem__(self, words_seq):
        self.hidden = self.init_hidden()
        return self.forward(words_seq)

    def forward(self, words):
        word_len = len(words)
        words = tuple(self.lang.ix(w) for w in words)
        words = to_tensor(words, dtype=torch.long)
        words = self.embedding(words)
        words = words.view(word_len, 1, -1)
        lstm_out, self.hidden = self.rnn(words, self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(words), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

    def save(self, name):
        torch.save(self, conf.path_in_zoo(name))

    @classmethod
    def load(cls, name):
        return torch.load(conf.path_in_zoo(name), map_location=lambda storage, loc: storage)

    def params_without_embed(self):
        for name, param in self.named_parameters():
            if 'embedding' not in name:
                yield param

    def init_params(self, pre_trained_wv=None):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        if pre_trained_wv is not None:
            device = self.embedding.weight.device
            wv_weight = self.embedding.weight.detach().cpu().numpy()
            self.lang.build_embedding(wv=pre_trained_wv, out_embedding=wv_weight)
            self.embedding.weight.data = torch.tensor(wv_weight, dtype=torch.float, device=device)


def to_tensor(value, dtype=torch.float32):
    if torch.is_tensor(value):
        return value.to(conf.device(), dtype=dtype)
    if isinstance(value, (list, tuple, np.ndarray)):
        return torch.tensor(value, dtype=dtype, device=conf.device())
    if isinstance(value, (float, np.float16, np.float32, np.float64)):
        return torch.tensor([value], dtype=dtype, device=conf.device())

    raise ValueError("Fail to convert {elem} to tensor".format(elem=value))
