import re
from jieba.posseg import cut as pseg_cut
from jieba import cut

NON_ASCII_PUNCTUATIONS = '\u2705✅．•·’丶《》—–－一¯§Ü▲◆◇※★☆♢▶☩▪＊❖Λ①②③④⑤⑥⑦⑧⑨²◎▼⊙■•●Ø⑴⑵⑶⑷㈡㈢°。､﹑、⃣'
NON_ASCII_PUNCTUATIONS_SET = set(NON_ASCII_PUNCTUATIONS)
HANS_DIGITS = '一二三四五六七八九十'
DIGITS = '0123456789一二三四五六七八九十'


def strips(s):
    return strip(s, '*&-_！） )。.；，,\n\t\r')


def remove_segment(s, beg, end):
    prev_beg_count = 0
    acc = []
    while s:
        ch = s[0]
        s = s[1:]
        if ch == beg:
            prev_beg_count += 1
        elif ch == end:
            if prev_beg_count > 0:
                prev_beg_count -= 1
            else:
                acc = []
        else:
            if prev_beg_count == 0:
                acc.append(ch)
    return ''.join(acc)


def remove_segments(s):
    return remove_segment(s, '(', ')')


def split_with(text, puncts):
    if not puncts:
        yield text
    else:
        for seg in text.split(puncts[0]):
            for item in split_with(seg, puncts[1:]):
                yield item


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


_regularize_punct_map = {
    '【': '[',
    '】': ']',
    '『': '"',
    '』': '"',
    '“': '"',
    '"': '"',
    '、': '､',
    '/': '､',
    '\t': ' ',
    '！': '!',
    '＂': '"',
    '＃': '#',
    '＄': '$',
    '％': '%',
    '＆': '&',
    '＇': '\'',
    '（': '(',
    '）': ')',
    '＊': '*',
    '＋': '+',
    '，': ',',
    '－': '-',
    '．': '.',
    '／': '/',
    '０': '0',
    '１': '1',
    '２': '2',
    '３': '3',
    '４': '4',
    '５': '5',
    '６': '6',
    '７': '7',
    '８': '8',
    '９': '9',
    '：': ':',
    '；': ';',
    '＜': '<',
    '＝': '=',
    '＞': '>',
    '？': '?',
    '＠': '@',
    'Ａ': 'A',
    'Ｂ': 'B',
    'Ｃ': 'C',
    'Ｄ': 'D',
    'Ｅ': 'E',
    'Ｆ': 'F',
    'Ｇ': 'G',
    'Ｈ': 'H',
    'Ｉ': 'I',
    'Ｊ': 'J',
    'Ｋ': 'K',
    'Ｌ': 'L',
    'Ｍ': 'M',
    'Ｎ': 'N',
    'Ｏ': 'O',
    'Ｐ': 'P',
    'Ｑ': 'Q',
    'Ｒ': 'R',
    'Ｓ': 'S',
    'Ｔ': 'T',
    'Ｕ': 'U',
    'Ｖ': 'V',
    'Ｗ': 'W',
    'Ｘ': 'X',
    'Ｙ': 'Y',
    'Ｚ': 'X',
    '［': '[',
    '＼': '\\',
    '］': ']',
    '＾': '^',
    '＿': '_',
    '｀': '`',
    'ａ': 'a',
    'ｂ': 'b',
    'ｃ': 'c',
    'ｄ': 'd',
    'ｅ': 'e',
    'ｆ': 'f',
    'ｇ': 'g',
    'ｈ': 'h',
    'ｉ': 'i',
    'ｊ': 'j',
    'ｋ': 'k',
    'ｌ': 'l',
    'ｍ': 'm',
    'ｎ': 'n',
    'ｏ': 'o',
    'ｐ': 'p',
    'ｑ': 'q',
    'ｒ': 'r',
    'ｓ': 's',
    'ｔ': 't',
    'ｕ': 'u',
    'ｖ': 'v',
    'ｗ': 'w',
    'ｘ': 'x',
    'ｙ': 'y',
    'ｚ': 'z',
    '｛': '{',
    '｜': '|',
    '｝': '}',
    '～': '~',
    '｟': '(',
    '｠': ')',
    '｢': '\'',
    '｣': '\'',
    '､': '､',
}


def regularize_punct(text):
    return ''.join([_regularize_punct_map.get(c, c.upper()) for c in text])


def regularize_coo_punct(text):
    """
    HR常常不会输入、常用逗号取代
    "HTML,Java" -> "HTML、Java"
    :param text:
    :return:
    """
    text_size = len(text)
    if text_size < 3:
        return text
    cseq = [text[0]]
    for i in range(1, text_size - 1):
        if text[i] == ',' and is_ascii(text[i - 1]) and is_ascii(text[i + 1]):
            cseq.append('､')
        else:
            cseq.append(text[i])
    cseq.append(text[-1])
    return ''.join(cseq)


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


def text_manip_util(text, str_op):
    """
    操作字符串，直到字符串不再变化为止
    :param text:字符串
    :param str_op:字符串操作
    :return:字符串
    """
    while True:
        new_text = str_op(text)
        if text == new_text:
            return text
        text = new_text


def trim_to_segments(text, specs, collect=None):
    def run_spec(t):
        ys, orig = [], ''
        while orig != t:
            orig = t
            for spec in specs:
                ys = [remove_redundant_space(seg) for seg in spec.execute(t, collect).split('\n')]
                ys = [x for x in ys if x]
                if len(ys) != 1:
                    return ys
                t = ys[0]
        return ys

    acc = [remove_redundant_space(text)]
    while acc:
        xs = run_spec(acc.pop())
        if len(xs) == 1:
            yield xs[0]
        else:
            acc.extend(xs)


def trim(text, specs, collect=None):
    def _do(t):
        t = remove_redundant_space(t)
        for spec in specs:
            t = remove_redundant_space(spec.execute(t, collect))
        return t

    return text_manip_util(text, _do)


def remove_postags(text, postag_set):
    def _run():
        for w in pseg_cut(text):
            if w.flag in postag_set:
                yield ' '
            else:
                yield w.word

    return remove_redundant_space(join_words2(_run()))


def trim_word_by(text, specs):
    def _trim():
        for w in pseg_cut(text):
            w = w.word
            for spec in specs:
                w = spec.execute(w)
            yield w

    return remove_redundant_space(join_words2(_trim()))


def translate_by(text, translator_dict):
    def _run():
        is_prev_translated = False
        for w in cut(text):
            if w in translator_dict:
                yield translator_dict[w]
                is_prev_translated = True
            elif w == ' ':
                if not is_prev_translated:
                    yield ' '
                is_prev_translated = False
            else:
                yield w
                is_prev_translated = False

    return remove_redundant_space(join_words2(_run()))


class TextTrim:
    def __init__(self, text):
        self.text = text

    def trim_by(self, specs):
        self.text = trim(self.text, specs)

    def remove_postags(self, postag_set):
        self.text = remove_postags(self.text, postag_set)

    def trim_word_by(self, specs):
        self.text = trim_word_by(self.text, specs)

    def translate_by(self, translator):
        self.text = translate_by(self.text, translator)

    def segment_by(self, sep):
        return re.split(sep, self.text)

    def __repr__(self):
        return self.text


def read_lines_from(file):
    with open(file, 'r') as f:
        for line in f:
            line = line.strip('\t\r\n ')
            if line:
                yield line



def contain_coo_punct(text):
    for w in pseg_cut(text):
        if w.word == '､' or w.flag == 'c':
            return True
    return False


def install_c_punct(text):
    def _run():
        for w in pseg_cut(text):
            word = '､' if w.flag == 'c' else w.word
            yield word

    return ''.join(_run())


def restore_c_punct(text):
    def _run():
        translated = False
        for c in reversed(text):
            if translated or c != '､':
                yield c
            else:
                translated = True
                yield '和'

    return ''.join(reversed(list(_run())))

