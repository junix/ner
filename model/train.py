import time
from itertools import islice
import torch
import torch.nn as nn
import torch.optim as optim

import jieba_dict
import conf
from dataset import generate_dataset
from model.lang import Lang
from .ner import EntityRecognizer, to_tensor

jieba_dict.init_user_dict()


class Metrics:
    def __init__(self, period=2000):
        self.started_at = int(time.time())
        self.ended_at = self.started_at
        self.loss = .0
        assert period > 0
        self.period = int(period)
        self.acc_count = 0

    def add_loss(self, loss):
        self.loss += float(loss)
        self.acc_count += 1
        self.ended_at = int(time.time())
        if self.acc_count % self.period == 0:
            print(self)
            self.loss, self.started_at = .0, self.ended_at

    def __str__(self):
        return '{count}:loss={loss:5.7},duration={elapsed}s'.format(
            count=self.acc_count,
            loss=self.loss,
            elapsed=self.ended_at - self.started_at)


def _make_period_saver(period, pkl_name):
    count = 0

    def _saver(model):
        nonlocal count
        count += 1
        if count % period == 0:
            model.save(pkl_name)
            print('save model:', pkl_name)

    return _saver


def _make_optimizer(optimizer_name, params, lr):
    if optimizer_name == 'sgd':
        return optim.SGD(params=params, lr=lr)
    elif optimizer_name == 'adam':
        return optim.Adam(params=params, lr=lr)
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(params=params, lr=lr)
    elif optimizer_name == 'adagrad':
        return optim.Adagrad(params=params, lr=lr)
    else:
        raise ValueError('not support optimizer:{}'.format(optimizer_name))


def _do_train(model, dataset, model_pkl_name, optimizer, lr):
    model.train()
    training_dataset = dataset
    criterion = nn.NLLLoss()
    optimizer = _make_optimizer(optimizer_name=optimizer, params=model.parameters(), lr=lr)
    saver = _make_period_saver(50000, pkl_name=model_pkl_name)
    metrics = Metrics()
    for sentence, target, faked in training_dataset:
        target = to_tensor(target, dtype=torch.long)
        model.zero_grad()
        model.hidden = model.init_hidden()
        tag_scores = model.forward(sentence)
        loss = criterion(tag_scores, target)
        loss.backward()
        optimizer.step()

        loss.detach_()
        metrics.add_loss(loss.item())
        saver(model)

    return model


def train_and_dump(from_model=None,
                   optimizer='sgd',
                   lr=1e-4,
                   rnn_type='lstm',
                   lang_pkl='lang.pt',
                   drop_n=0,
                   real_corpus_sample=0.3):
    if from_model:
        model = EntityRecognizer.load(from_model)
        model_pkl_name = from_model
    else:
        lang = Lang.load(conf.path_in_zoo(lang_pkl))
        model = EntityRecognizer(lang=lang, embedding_dim=200, rnn_type=rnn_type)
        model.init_params()
        model_pkl_name = 'model.{rnn_type}.{optimizer}'.format(rnn_type=rnn_type, optimizer=optimizer)
    model.move_to_device(conf.device())
    dataset = islice(generate_dataset(real_corpus_sample), drop_n, None)
    _do_train(model, dataset, model_pkl_name, optimizer=optimizer, lr=lr)
