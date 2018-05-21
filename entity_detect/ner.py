import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import jieba_dict
from conf import DEVICE


class EntityRecognizer(nn.Module):

    def __init__(self, input_size=-1, num_layers=2, hidden_size=2048):
        super(EntityRecognizer, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, dropout=0.5, num_layers=self.num_layers)
        # self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.hidden2tag = nn.Linear(hidden_size, out_features=3)
        self.hidden = self.init_hidden()
        self.move_to_device(DEVICE)

    def move_to_device(self, device):
        if device.type == 'cpu':
            self.cpu()
        else:
            self.cuda()

    def init_hidden(self):
        # return to_var(torch.zeros(self.num_layers, 1, self.hidden_size), self.use_gpu)
        return (
            torch.zeros(self.num_layers, 1, self.hidden_size, device=DEVICE),
            torch.zeros(self.num_layers, 1, self.hidden_size, device=DEVICE)
        )

    def __getitem__(self, words_seq):
        self.hidden = self.init_hidden()
        return self.forward(words_seq)

    def forward(self, words):
        words = to_tensor(words)
        word_len = len(words)
        words = words.view(word_len, 1, -1)
        lstm_out, self.hidden = self.rnn(words, self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(words), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

    def save(self, file):
        torch.save(self, file)


def to_tensor(value):
    if torch.is_tensor(value):
        return value.to(DEVICE)
    if isinstance(value, (list, tuple, np.ndarray)):
        return torch.tensor(value, dtype=torch.float32, device=DEVICE)
    if isinstance(value, (float, np.float16, np.float32, np.float64)):
        return torch.tensor([value], dtype=torch.float32, device=DEVICE)

    raise ValueError("Fail to convert {elem} to tensor".format(elem=value))
