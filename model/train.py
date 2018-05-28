import time
import torch
import torch.nn as nn
import torch.optim as optim

import jieba_dict
from conf import DEVICE
from dataset import generate_dataset
from dataset.lang import Lang
from .ner import EntityRecognizer, to_tensor

jieba_dict.init_user_dict()


class Metrics:
    def __init__(self, period=2000):
        self.started_at = int(time.time())
        self.loss = .0
        assert period > 0
        self.period = int(period)
        self.acc_count = 0

    def add_loss(self, loss):
        self.loss += float(loss)
        self.acc_count += 1
        if self.acc_count % self.period == 0:
            ended_at = int(time.time())
            print(self.acc_count, '=>', self.loss, 'time duration =>', ended_at - self.started_at)
            self.loss, self.started_at = .0, ended_at


def _make_period_saver(period, dump_name):
    count = 0

    def _saver(model):
        nonlocal count
        count += 1
        if count % period == 0:
            print('save model', dump_name)
            model.save(dump_name)

    return _saver


def do_train(model, dataset, lang, dump_name, lr):
    model.train()
    training_dataset = dataset
    loss_function = nn.NLLLoss()
    optimizer_for_real = optim.SGD(model.parameters(), lr=lr)
    # optimizer_for_fake = optim.SGD(model.params_without_embed(), lr=lr)
    saver = _make_period_saver(50000, dump_name=dump_name)
    metrics = Metrics()
    for sentence, target, faked in training_dataset:
        sentence = lang.to_index(sentence)
        sentence = to_tensor(sentence, dtype=torch.long)
        target = to_tensor(target, dtype=torch.long)
        model.zero_grad()
        model.hidden = model.init_hidden()
        tag_scores = model.forward(sentence)
        loss = loss_function(tag_scores, target)
        loss.backward()
        # if faked:
        #     optimizer_for_fake.step()
        # else:
        optimizer_for_real.step()
        metrics.add_loss(loss.item())
        saver(model)

    return model


def train_and_dump(drop_n=0, from_model=None, new_rnn_type='lstm', lr=1e-4):
    lang = Lang.load()
    if from_model:
        model = EntityRecognizer.load(from_model)
        dump_model_name = from_model
    else:
        model = EntityRecognizer(vocab_size=lang.vocab_size(), embedding_dim=200, rnn_type=new_rnn_type)
        model.init_params()
        dump_model_name = 'model.{rnn_type}'.format(rnn_type=new_rnn_type)
    model.move_to_device(DEVICE)
    dataset = generate_dataset(drop_n=drop_n)
    do_train(model, dataset, lang, dump_model_name, lr=lr)
