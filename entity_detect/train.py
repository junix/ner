import torch
import torch.nn as nn
import torch.optim as optim

import jieba_dict
from conf import DEVICE
from dataset.gen_dataset import generate_dataset
from .ner import EntityRecognizer, to_tensor
from conf import MODEL_DUMP_FILE
from dataset.lang import Lang

jieba_dict.init_user_dict()


def train(model, dataset, lang):
    model.train()
    training_dataset = dataset
    loss_function = nn.NLLLoss()
    lr = 1e-3
    optimizer_for_real = optim.RMSprop(model.parameters(), lr=lr)
    # optimizer_for_fake = optim.SGD(model.params_without_embed(), lr=lr)
    count, acc_loss = 0, 0.0
    for epoch in range(60):
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
            acc_loss += loss.item()
            count += 1
            if count % 2000 == 0:
                print(count, ' => ', float(acc_loss))
                acc_loss = 0
            if count % 50000 == 0:
                model.save(MODEL_DUMP_FILE)

    return model


def train_and_dump(load_old=False):
    dataset = generate_dataset()
    lang = Lang.load()
    if load_old:
        model = torch.load(MODEL_DUMP_FILE)
    else:
        model = EntityRecognizer(input_size=200, vocab_size=lang.vocab_size())
        model.init_params()
    model.move_to_device(DEVICE)
    train(model, dataset, lang)


def detect_input_shape(dataset):
    for x, _ in dataset:
        return x.shape[-1]
    raise ValueError('empty dataset')
