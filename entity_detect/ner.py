import os
import re

import jieba
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import dataset.transformer as transformer
import jieba_dict
from dataset.gen_dataset import load_dataset
from utils.str_algo import regularize_punct

_model_dump_dir = '{pwd}{sep}..{sep}model_dump'.format(
    pwd=os.path.dirname(__file__), sep=os.path.sep)
_default_model_dump_file = _model_dump_dir + os.path.sep + 'model.dump'

jieba_dict.init_user_dict()


class EntityRecognizer(nn.Module):

    def __init__(self, device, input_size=-1, num_layers=2, hidden_size=1024):
        super(EntityRecognizer, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, dropout=0.5, num_layers=self.num_layers)
        # self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.hidden2tag = nn.Linear(hidden_size, out_features=3)
        self.hidden = self.init_hidden()
        self.change_context(device)

    def change_context(self, device):
        self.device = device
        if self.device.type == 'cpu':
            self.cpu()
        else:
            self.cuda()

    def init_hidden(self):
        # return to_var(torch.zeros(self.num_layers, 1, self.hidden_size), self.use_gpu)
        return (
            torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device),
            torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device)
        )

    def __getitem__(self, words_seq):
        self.hidden = self.init_hidden()
        return self.forward(words_seq)

    def forward(self, words):
        words = to_tensor(words, self.device)
        word_len = len(words)
        words = words.view(word_len, 1, -1)
        lstm_out, self.hidden = self.rnn(words, self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(words), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

    def save(self, file):
        orig_device, self.device = self.device, None
        torch.save(self, file)
        self.device = orig_device


def to_tensor(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, (list, np.ndarray)):
        return torch.tensor(value, dtype=torch.float32, device=device)
    if isinstance(value, (float, np.float16, np.float32, np.float64)):
        return torch.tensor([value], dtype=torch.float32, device=device)

    raise ValueError("Fail to convert {elem} to tensor".format(elem=value))


def train(model, dataset):
    model.train()
    count = 1
    training_dataset = dataset
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    for epoch in range(60):
        for sentence, target in training_dataset:
            sentence = to_tensor(sentence, model.device)
            target = to_tensor(target, model.device).long()
            model.zero_grad()
            model.hidden = model.init_hidden()
            tag_scores = model.forward(sentence)
            loss = loss_function(tag_scores, target)
            loss.backward()
            optimizer.step()
            count += 1
            if count % 10000 == 0:
                print('processed sentences count = ', count)
            if count % 50000 == 0:
                model.save(_default_model_dump_file)

    return model


def train_and_dump(load_old=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = load_dataset()
    if load_old:
        model = torch.load(_default_model_dump_file)
        model.change_context(device)
    else:
        model = EntityRecognizer(device=device, input_size=detect_input_shape(dataset))
    train(model, dataset)


def detect_input_shape(dataset):
    for x, _ in dataset:
        return x.shape[-1]
    raise ValueError('empty dataset')


def fetch_tags(model, text):
    words = list(jieba.cut(text))
    sentence = transformer.transform(words)

    with torch.no_grad():
        output = model[sentence]
        _, tags = output.max(dim=1)
        return words, tags.tolist()


def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(_default_model_dump_file, map_location=lambda storage, loc: storage)
    model.eval()
    model.change_context(device)
    return model


def load_predict(output_keyword=False):
    model = load_model()

    def predict(sentence):
        sentence = regularize_punct(sentence)
        if not sentence:
            return '' if output_keyword else []
        if not output_keyword:
            _words, tags = fetch_tags(model, sentence)
            return tags
        sub_sentences = re.split('[,.!?]', sentence)
        old_category, old_keywords = '', ''
        for sub_sentence in sub_sentences:
            words, tags = fetch_tags(model, sub_sentence)
            category, keywords = select_keywords(words, tags)
            if category and keywords:
                return category, keywords
            if not old_category and category:
                old_category = category
            if not old_keywords and keywords:
                old_keywords = keywords

        if not old_keywords:
            old_keywords = sentence
        return old_category, old_keywords

    return predict


def select_keywords(words, tags):
    keywords, category, prev_tag = [], [], 0
    for word, tag in zip(words, tags):
        if tag == 1:
            if prev_tag != 1 and keywords:
                keywords.append(' ')
            keywords.append(word)
        elif tag == 2:
            if prev_tag != 2 and category:
                category.append(' ')
            category.append(word)
        prev_tag = tag
    return ''.join(category), ''.join(keywords)
