_replace_map = {
    '乐彩': '乐才',
    '乐采': '乐才',
    '小了': '小乐',
    '九零后': '90后',
    '八零后': '80后',
    '七零后': '70后'
}


def replace_to_common_words(words):
    for w in words:
        yield _replace_map.get(w, w)
