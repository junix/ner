def remove_stopwords(words):
    import jieba
    from yxt_nlp.utils import is_stopword
    from itertools import dropwhile
    if isinstance(words, (list, tuple)):
        words.reverse()
        words = list(dropwhile(lambda x: is_stopword(x), words))
        words.reverse()
        words = list(dropwhile(lambda x: is_stopword(x), words))
        return words
    elif isinstance(words, str):
        words = list(jieba.cut(words))
        words = remove_stopwords(words)
        return ''.join(words)

    raise ValueError("{} can't remove stopwords".format(words))
