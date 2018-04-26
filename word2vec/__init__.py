from .rpc import Word2Vec

wv = Word2Vec()


def switch_to_local_gensim():
    global wv
    from .gensims import Word2Vec as WV
    wv = WV()
