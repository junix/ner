import re
from jieba.posseg import cut as pseg_cut
from jieba import cut

NON_ASCII_PUNCTUATIONS = '\u2705✅．•·’丶《》—–－一¯§Ü▲◆◇※★☆♢▶☩▪＊❖Λ①②③④⑤⑥⑦⑧⑨²◎▼⊙■•●Ø⑴⑵⑶⑷㈡㈢°。､﹑、⃣'
NON_ASCII_PUNCTUATIONS_SET = set(NON_ASCII_PUNCTUATIONS)
HANS_DIGITS = '一二三四五六七八九十'
DIGITS = '0123456789一二三四五六七八九十'


def strip(text, trim_chars):
    if not text:
        return text
    if text[0] in trim_chars:
        return strip(text[1:], trim_chars)
    elif text[-1] in trim_chars:
        return strip(text[:-1], trim_chars)
    else:
        return text


def non_alphanum_ascii_chars():
    for c in range(180):
        ch = chr(c)
        if ('0' <= ch <= '9') or ('a' <= ch <= 'z') or ('A' <= ch <= 'Z'):
            continue
        yield ch


def end_with(text, predicate):
    if not text:
        return False
    return predicate(text[-1])


def start_with(text, predicate):
    if not text:
        return False
    return predicate(text[0])


def join_words(words):
    sentence = ''
    for word in words:
        need_space = end_with(sentence, lambda x: ord(x) < 128) and \
                     start_with(word, lambda x: ord(x) < 128)
        delim = ' ' if need_space else ''
        sentence = sentence + delim + word
    return sentence


def join_words2(words):
    sentence = ''
    for word in words:
        need_space = end_with(sentence, is_ascii_alphanum) and \
                     start_with(word, is_ascii_alphanum)
        delim = ' ' if need_space else ''
        sentence = sentence + delim + word
    return sentence


def is_ascii(ch):
    return ord(ch) < 128


def is_ascii_text(text):
    for ch in text:
        if (not is_ascii(ch)) and (ch not in NON_ASCII_PUNCTUATIONS_SET):
            return False
    return True


def is_han_char(ch):
    return '\u4E00' <= ch <= '\u9FFF'


def is_hans_text(text):
    return not is_ascii_text(text)


def is_ascii_alpha(ch):
    return 'a' <= ch <= 'z' or 'A' <= ch <= 'Z'


def is_ascii_num(ch):
    return '0' <= ch <= '9'


def is_ascii_alphanum(ch):
    return is_ascii_alpha(ch) or is_ascii_num(ch)


_trim_multi_space_regex = re.compile(r' {2,}')


def remove_redundant_space(text):
    """
    删除多余的字符串：
    1. 开头和结尾的空格
    2. 合并多个连续空格为1个空格
    :param text:
    :return:
    """
    return _trim_multi_space_regex.sub(' ', text.strip())


def read_lines_from(file):
    with open(file, 'r') as f:
        for line in f:
            line = line.strip('\t\r\n ')
            if line:
                yield line
