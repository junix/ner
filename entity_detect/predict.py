import re

import torch
import jieba
from regularize.replace import replace_to_common_words, regularize_punct
from regularize.remove_stopwords import remove_stopwords
from utils.str_algo import is_ascii_text
import jieba_dict
from dataset.lang import Lang

jieba_dict.init_user_dict()


def fetch_tags(model, text, lang):
    words = list(replace_to_common_words(jieba.cut(text)))
    sentence = lang.to_index(words)

    with torch.no_grad():
        output = model[sentence]
        _, tags = output.max(dim=1)
        return words, tags.tolist()


def load_model():
    from conf import DEVICE, MODEL_DUMP_FILE
    model = torch.load(MODEL_DUMP_FILE, map_location=lambda storage, loc: storage)
    model.eval()
    model.move_to_device(DEVICE)
    return model


def load_predict(output_keyword=False):
    model = load_model()
    lang = Lang.load()

    def predict(sentence):
        sentence = regularize_punct(sentence)
        if not sentence:
            return '' if output_keyword else []
        if not output_keyword:
            _words, tags = fetch_tags(model, sentence, lang)
            return tags
        if is_ascii_text(sentence):
            return '', sentence
        sub_sentences = re.split('[,.!?]', sentence)
        old_category, old_keywords = '', ''
        all_words = []
        for sub_sentence in sub_sentences:
            words, tags = fetch_tags(model, sub_sentence, lang)
            category, keywords = select_keywords(words, tags)
            if len(old_category) < len(category):
                old_category = category
            if len(old_keywords) < len(keywords):
                old_keywords = keywords
            if not old_keywords:
                all_words.extend(words)

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
