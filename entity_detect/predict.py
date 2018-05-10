import re

import torch



def fetch_tags(model, text):
    import jieba
    import dataset.transformer as transformer
    words = list(jieba.cut(text))
    sentence = transformer.transform(words)

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
    from utils.str_algo import regularize_punct
    import jieba_dict
    model = load_model()

    def predict(sentence):
        jieba_dict.init_user_dict()
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
