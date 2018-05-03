import torch
import torch.nn as nn
import torch.optim as optim

import jieba_dict
from conf import DEVICE
from dataset.gen_dataset import load_dataset
from .ner import EntityRecognizer, to_tensor
from conf import MODEL_DUMP_FILE

jieba_dict.init_user_dict()


def train(model, dataset):
    model.train()
    count = 1
    training_dataset = dataset
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    for epoch in range(60):
        for sentence, target in training_dataset:
            sentence = to_tensor(sentence)
            target = to_tensor(target).long()
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
                model.save(MODEL_DUMP_FILE)

    return model


def train_and_dump(load_old=False):
    dataset = load_dataset()
    if load_old:
        model = torch.load(MODEL_DUMP_FILE)
    else:
        model = EntityRecognizer(input_size=detect_input_shape(dataset))
    model.move_to_device(DEVICE)
    train(model, dataset)


def detect_input_shape(dataset):
    for x, _ in dataset:
        return x.shape[-1]
    raise ValueError('empty dataset')
