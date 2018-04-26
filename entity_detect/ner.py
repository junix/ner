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
from dataset.gen_dataset import generate_dataset
from utils.str_algo import regularize_punct

_model_dump_dir = '{pwd}{sep}..{sep}model_dump'.format(
    pwd=os.path.dirname(__file__), sep=os.path.sep)
_default_model_dump_file = _model_dump_dir + os.path.sep + 'model.dump'

jieba_dict.init_user_dict()


class EntityRecognizer(nn.Module):

    def __init__(self, device, input_size=-1, num_layers=2, hidden_size=256):
        super(EntityRecognizer, self).__init__()
        self.device = device
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


def to_tensor(val, device):
    if isinstance(val, torch.Tensor):
        return val.to(device)
    if isinstance(val, np.ndarray):
        tensor = torch.from_numpy(val.astype(np.float32))
        return tensor.to(device)
    if isinstance(val, (float, np.float32, np.float64, np.float16)):
        tensor = torch.FloatTensor([float(val)])
        return tensor.to(device)

    raise TypeError("Fail to convert {elem} to tensor".format(elem=val))


def train(model, dataset):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_function = nn.NLLLoss()

    # model.change_device(device)
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
            sentence = to_tensor(sentence, model.device)
            target = to_tensor(target, model.device).long()
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


def train_and_dump(load_old=False):
    import word2vec
    # use local gensim to accelerate training
    word2vec.wv = word2vec.gensims.Word2Vec()
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
    output = model[sentence]
    output.detach_()
    # tags = output.detach().cpu().numpy().argmax(axis=1)
    _, tags = output.max(dim=1)
    return words, tags.tolist()


def load_predict(model=None, output_keyword=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model is None:
        model = torch.load(_default_model_dump_file, map_location=lambda storage, loc: storage)
        model.eval()
        model.change_context(device)

    def predict(sentence):
        sentence = regularize_punct(sentence)
        if not sentence:
            return '' if output_keyword else []
        if not output_keyword:
            _words, tags = fetch_tags(model, sentence)
            return tags
        sub_sentences = re.split('[,.!?]', sentence)
        for sub_sentence in sub_sentences:
            words, tags = fetch_tags(model, sub_sentence)
            keywords = select_keywords(words, tags)
            if keywords:
                return keywords
        return sentence

    return predict


def select_keywords(words, tags):
    keywords, prev_tag = [], 0
    for word, tag in zip(words, tags):
        if tag == 1:
            if prev_tag != 1 and keywords:
                keywords.append(' ')
            keywords.append(word)
        prev_tag = tag
    return ''.join(keywords)
