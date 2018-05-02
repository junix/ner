import torch
import torch.nn as nn

from conf import DEVICE


class Encoder(nn.Module):

    def __init__(self, input_size=200, hidden_size=256):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, word, hidden):
        word = word.view(1, 1, -1)
        output, hidden = self.gru(word, hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)
