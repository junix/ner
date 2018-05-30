import re

import torch
import jieba

from yxt_nlp.utils import jieba_load_userdict, is_ascii_text, regularize_punct

from regularize import replace_to_common_words, remove_stopwords
import conf
from .ner import EntityRecognizer

jieba_load_userdict()


def fetch_tags(model, text):
    words = tuple(replace_to_common_words(jieba.cut(text)))
    with torch.no_grad():
        output = model[words]
        _, tags = output.max(dim=1)
        return words, tags.tolist()


def load_model(model_name):
    model = EntityRecognizer.load(model_name)
    model.eval()
    model.move_to_device(conf.device())
    return model


def load_predict(model_name='model.pt', output_keyword=False):
    model = load_model(model_name)

    def predict(sentence):
        sentence = regularize_punct(sentence)
        if not sentence:
            return '' if output_keyword else []
        if not output_keyword:
            _words, tags = fetch_tags(model, sentence)
            return tags
        if is_ascii_text(sentence):
            return '', sentence
        sub_sentences = re.split('[,.。;!?]', sentence)
        old_category, old_keywords = '', ''
        all_words = []
        for sub_sentence in sub_sentences:
            words, tags = fetch_tags(model, sub_sentence)
            category, keywords = select_keywords(words, tags)
            if category and keywords:
                old_category, old_keywords = category, keywords
                break
            if len(old_category) < len(category):
                old_category = category
            if len(old_keywords) < len(keywords):
                old_keywords = keywords
            if not old_keywords:
                all_words.extend(words)
            if category and keywords:
                break

        if not old_keywords:
            old_keywords = ''.join(remove_stopwords(all_words))
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
    keywords = remove_stopwords(keywords)
    return ''.join(category), ''.join(keywords)
