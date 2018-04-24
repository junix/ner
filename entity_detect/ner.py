import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import jieba
from torch.autograd import Variable

import dataset.transformer as transformer
from dataset.gen_dataset import generate_dataset

_model_dump_dir = '{pwd}{sep}..{sep}model_dump'.format(
    pwd=os.path.dirname(__file__), sep=os.path.sep)
_default_transformer_dump_file = _model_dump_dir + os.path.sep + 'transformer.pickle'
_default_model_dump_file = _model_dump_dir + os.path.sep + 'model.dump'


class EntityRecognizer(nn.Module):

    def __init__(self, input_size=-1, num_layers=1, hidden_size=256):
        super(EntityRecognizer, self).__init__()
        self.use_gpu = False
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, dropout=0.5, num_layers=self.num_layers)
        self.relu = nn.ReLU()
        # self.bn = nn.BatchNorm1d(hidden_size)
        # self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size)
        # self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.hidden2tag = nn.Linear(hidden_size, out_features=2)
        self.hidden = self.init_hidden()

    def in_cpu_context(self):
        self.use_gpu = False
        self.cpu()

    def in_cuda_context(self):
        self.use_gpu = True
        self.cuda()

    def change_context(self, use_gpu):
        if use_gpu:
            self.in_cuda_context()
        else:
            self.in_cpu_context()

    def init_hidden(self):
        # return to_var(torch.zeros(self.num_layers, 1, self.hidden_size), self.use_gpu)
        return (
            to_var(torch.zeros(self.num_layers, 1, self.hidden_size), self.use_gpu),
            to_var(torch.zeros(self.num_layers, 1, self.hidden_size), self.use_gpu),
        )

    def __getitem__(self, words_seq):
        self.hidden = self.init_hidden()
        return self.forward(words_seq)

    def forward(self, words):
        words = to_var(words, self.use_gpu)
        word_len = len(words)
        words = words.view(word_len, 1, -1)
        lstm_out, self.hidden = self.rnn(words, self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(words), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

    def save(self, file):
        torch.save(self, file)


def to_var(val, use_gpu=False):
    def _as_var():
        if isinstance(val, torch.autograd.variable.Variable):
            return val
        if isinstance(val, np.ndarray):
            tensor = torch.from_numpy(val.astype(np.float32))
            return Variable(tensor.float())
        if isinstance(val, (float, np.float32, np.float64, np.float16)):
            return Variable(torch.FloatTensor([float(val)]))
        if isinstance(val, (torch.FloatTensor, torch.DoubleTensor)):
            return Variable(val)
        raise TypeError("{elem} is not ndarray".format(elem=val))

    if use_gpu:
        return _as_var().cuda()
    else:
        return _as_var().cpu()


def train(model, dataset, use_gpu):
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_function = nn.NLLLoss()

    model.change_context(use_gpu)
    # See what the scores are before training
    # Note that element i,j of the output is the score for ner j for word i.

    # testset_size = 10000
    # ys_test = ys[:testset_size]
    # xs = [to_var(x, use_gpu=use_gpu) for x in xs]
    # ys = [to_var(y, use_gpu=use_gpu) for y in ys]
    # xs_test = xs[:testset_size]
    # training_dataset = [(to_var(x, use_gpu), to_var(y, use_gpu).long()) for x, y in dataset]
    training_dataset = dataset

    # def accu():
    #     ys_hat = F.sigmoid(torch.cat([model[x] for x in xs_test]))
    #     ys_hat = ys_hat.data.cpu().numpy()
    #     ys_hat = ys_hat.reshape(ys_test.shape)
    #     ys_hat = np.where(ys_hat > 0.5, 1.0, .0)
    #     return accuracy_score(ys_test, ys_hat)

    count = 1
    for epoch in range(60):  # again, normally you would NOT do 300 epochs, it is toy data
        for sentence, target in training_dataset:
            sentence = to_var(sentence, use_gpu)
            target = to_var(target, use_gpu).long()
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Also, we need to clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            model.hidden = model.init_hidden()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Variables of word indices.

            # targets = prepare_sequence(tags, tag_to_ix)
            # Step 3. Run our forward pass.
            tag_scores = model.forward(sentence)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            # target_in = target_in.view(tag_scores.shape)
            # tag_scores = tag_scores.view(target.shape)
            # loss = F.binary_cross_entropy(tag_scores, target)
            loss = loss_function(tag_scores, target)
            loss.backward()
            optimizer.step()
            count += 1
            if count % 10000 == 0:
                print('processed sentences count = ', count)
            if count % 50000 == 0:
                model.save(_default_model_dump_file)

    return model


def load_dataset():
    dataset = generate_dataset()
    for xs, y_true in dataset:
        yield transformer.transform(xs), np.array(y_true)


def train_and_dump(load_old=False, use_gpu=False):
    dataset = load_dataset()
    if load_old:
        model = torch.load(_default_model_dump_file)
    else:
        # TODO:
        model = EntityRecognizer(input_size=200)
    train(model, dataset, use_gpu=use_gpu)


def load_predict(model=None, use_gpu=False, output_keyword=False):
    if model is None:
        model = torch.load(_default_model_dump_file, map_location=lambda storage, loc: storage)
        model.change_context(use_gpu)

    def predict(sentence):
        if not sentence:
            return ''
        words = list(jieba.cut(sentence))
        input = transformer.transform(words)
        output = model[input]
        tags = output.data.numpy().argmax(axis=1)
        if not output_keyword:
            return tags

        phrase = []
        for word, tag in zip(words, tags):
            if tag == 1:
                phrase.append(word)
            else:
                if phrase:
                    return ''.join(phrase)
        best = ''.join(phrase)
        if best:
            return best
        return sentence

    return predict

