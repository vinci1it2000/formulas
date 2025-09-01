#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2025 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
Python equivalents of text Excel functions.
"""
import re
import json
import regex
import functools
import unicodedata
import numpy as np
import schedula as sh
from ..errors import FoundError
from . import (
    wrap_ufunc, Error, replace_empty, XlError, flatten, wrap_func, is_not_empty,
    raise_errors, _text2num, Array, get_error, return_2d_func
)

FUNCTIONS = {}
_kw0 = {
    'input_parser': lambda *a: a,
    'args_parser': lambda *a: map(functools.partial(replace_empty, empty=''), a)
}
codes = {
    1: "\x01",
    2: "\x02",
    3: "\x03",
    4: "\x04",
    5: "\x05",
    6: "\x06",
    7: "\x07",
    8: "\x08",
    9: "\t",
    10: "\n",
    11: "\x0B",
    12: "\x0C",
    13: "\x0D",
    14: "\x0E",
    15: "\x0F",
    16: "\x10",
    17: "\x11",
    18: "\x12",
    19: "\x13",
    20: "\x14",
    21: "\x15",
    22: "\x16",
    23: "\x17",
    24: "\x18",
    25: "\x19",
    26: "\x1A",
    27: "\x1B",
    28: "\x1C",
    29: "\x1D",
    30: "\x1E",
    31: "\x1F",
    32: " ",
    33: "!",
    34: '"',
    35: "#",
    36: "$",
    37: "%",
    38: "&",
    39: "'",
    40: "(",
    41: ")",
    42: "*",
    43: "+",
    44: ",",
    45: "-",
    46: ".",
    47: "/",
    48: "0",
    49: "1",
    50: "2",
    51: "3",
    52: "4",
    53: "5",
    54: "6",
    55: "7",
    56: "8",
    57: "9",
    58: ":",
    59: ";",
    60: "<",
    61: "=",
    62: ">",
    63: "?",
    64: "@",
    65: "A",
    66: "B",
    67: "C",
    68: "D",
    69: "E",
    70: "F",
    71: "G",
    72: "H",
    73: "I",
    74: "J",
    75: "K",
    76: "L",
    77: "M",
    78: "N",
    79: "O",
    80: "P",
    81: "Q",
    82: "R",
    83: "S",
    84: "T",
    85: "U",
    86: "V",
    87: "W",
    88: "X",
    89: "Y",
    90: "Z",
    91: "[",
    92: "\\",
    93: "]",
    94: "^",
    95: "_",
    96: "`",
    97: "a",
    98: "b",
    99: "c",
    100: "d",
    101: "e",
    102: "f",
    103: "g",
    104: "h",
    105: "i",
    106: "j",
    107: "k",
    108: "l",
    109: "m",
    110: "n",
    111: "o",
    112: "p",
    113: "q",
    114: "r",
    115: "s",
    116: "t",
    117: "u",
    118: "v",
    119: "w",
    120: "x",
    121: "y",
    122: "z",
    123: "{",
    124: "|",
    125: "}",
    126: "~",
    127: "",
    128: "Ä",
    129: "Å",
    130: "Ç",
    131: "É",
    132: "Ñ",
    133: "Ö",
    134: "Ü",
    135: "á",
    136: "à",
    137: "â",
    138: "ä",
    139: "ã",
    140: "å",
    141: "ç",
    142: "é",
    143: "è",
    144: "ê",
    145: "ë",
    146: "í",
    147: "ì",
    148: "î",
    149: "ï",
    150: "ñ",
    151: "ó",
    152: "ò",
    153: "ô",
    154: "ö",
    155: "õ",
    156: "ú",
    157: "ù",
    158: "û",
    159: "ü",
    160: "†",
    161: "°",
    162: "¢",
    163: "£",
    164: "§",
    165: "•",
    166: "¶",
    167: "ß",
    168: "®",
    169: "©",
    170: "™",
    171: "´",
    172: "¨",
    173: "≠",
    174: "Æ",
    175: "Ø",
    176: "∞",
    177: "±",
    178: "≤",
    179: "≥",
    180: "¥",
    181: "µ",
    182: "∂",
    183: "∑",
    184: "∏",
    185: "π",
    186: "∫",
    187: "ª",
    188: "º",
    189: "Ω",
    190: "æ",
    191: "ø",
    192: "¿",
    193: "¡",
    194: "¬",
    195: "√",
    196: "ƒ",
    197: "≈",
    198: "∆",
    199: "«",
    200: "»",
    201: "…",
    202: " ",
    203: "À",
    204: "Ã",
    205: "Õ",
    206: "Œ",
    207: "œ",
    208: "–",
    209: "—",
    210: "“",
    211: "”",
    212: "‘",
    213: "’",
    214: "÷",
    215: "◊",
    216: "ÿ",
    217: "Ÿ",
    218: "⁄",
    219: "€",
    220: "‹",
    221: "›",
    222: "ﬁ",
    223: "ﬂ",
    224: "‡",
    225: "·",
    226: "‚",
    227: "„",
    228: "‰",
    229: "Â",
    230: "Ê",
    231: "Á",
    232: "Ë",
    233: "È",
    234: "Í",
    235: "Î",
    236: "Ï",
    237: "Ì",
    238: "Ó",
    239: "Ô",
    240: "",
    241: "Ò",
    242: "Ú",
    243: "Û",
    244: "Ù",
    245: "ı",
    246: "ˆ",
    247: "˜",
    248: "¯",
    249: "˘",
    250: "˙",
    251: "˚",
    252: "¸",
    253: "˝",
    254: "˛",
    255: "ˇ"
}
inverse_codes = {v: k for k, v in codes.items()}

# Full-width ASCII block → ASCII (plus ideographic space)
_FULLWIDTH_ASCII = {chr(cp): chr(cp - 0xFEE0) for cp in range(0xFF01, 0xFF5F)}
_FULLWIDTH_ASCII['\u3000'] = ' '  # ideographic space → space

# Base Katakana → Halfwidth Katakana
_KATAKANA_BASE_TO_HALF = {
    'ァ': 'ｧ', 'ア': 'ｱ', 'ィ': 'ｨ', 'イ': 'ｲ', 'ゥ': 'ｩ', 'ウ': 'ｳ', 'ェ': 'ｪ',
    'エ': 'ｴ', 'ォ': 'ｫ', 'オ': 'ｵ',
    'カ': 'ｶ', 'キ': 'ｷ', 'ク': 'ｸ', 'ケ': 'ｹ', 'コ': 'ｺ', 'サ': 'ｻ', 'シ': 'ｼ',
    'ス': 'ｽ', 'セ': 'ｾ', 'ソ': 'ｿ',
    'タ': 'ﾀ', 'チ': 'ﾁ', 'ツ': 'ﾂ', 'テ': 'ﾃ', 'ト': 'ﾄ', 'ナ': 'ﾅ', 'ニ': 'ﾆ',
    'ヌ': 'ﾇ', 'ネ': 'ﾈ', 'ノ': 'ﾉ',
    'ハ': 'ﾊ', 'ヒ': 'ﾋ', 'フ': 'ﾌ', 'ヘ': 'ﾍ', 'ホ': 'ﾎ', 'マ': 'ﾏ', 'ミ': 'ﾐ',
    'ム': 'ﾑ', 'メ': 'ﾒ', 'モ': 'ﾓ',
    'ヤ': 'ﾔ', 'ユ': 'ﾕ', 'ヨ': 'ﾖ', 'ラ': 'ﾗ', 'リ': 'ﾘ', 'ル': 'ﾙ', 'レ': 'ﾚ',
    'ロ': 'ﾛ', 'ワ': 'ﾜ', 'ヲ': 'ｦ', 'ン': 'ﾝ',
    'ャ': 'ｬ', 'ュ': 'ｭ', 'ョ': 'ｮ', 'ッ': 'ｯ',
    # Small kana & specials
    'ヵ': 'ｶ', 'ヶ': 'ｹ',
    '・': '･', 'ー': 'ｰ',  # punctuation
    '。': '｡', '、': '､', '「': '｢', '」': '｣',
}

# Combining voice marks → Halfwidth voice marks
_COMB_DAKUTEN = '\u3099'  # ◌゙
_COMB_HANDAKUTEN = '\u309A'  # ◌゚
_HALF_DAKUTEN = 'ﾞ'  # FF9E
_HALF_HANDAKUTEN = 'ﾟ'  # FF9F


def xasc(value):
    """
    Excel-like ASC: convert full-width ASCII & Katakana to half-width.
    Leaves hiragana/kanji unchanged.
    """
    s = _str(value)

    out = []
    for ch in s:
        # Fast path: full-width ASCII / punctuation
        if ch in _FULLWIDTH_ASCII:
            out.append(_FULLWIDTH_ASCII[ch])
            continue

        # Decompose to catch precomposed voiced kana (e.g., ガ = カ + ◌゙)
        decomp = unicodedata.normalize('NFD', ch)

        # If no change after NFD, decomp is typically just ch
        for d in decomp:
            if d in _KATAKANA_BASE_TO_HALF:
                out.append(_KATAKANA_BASE_TO_HALF[d])
            elif d == _COMB_DAKUTEN:
                out.append(_HALF_DAKUTEN)
            elif d == _COMB_HANDAKUTEN:
                out.append(_HALF_HANDAKUTEN)
            else:
                # Not a target for conversion: keep as-is
                out.append(d)

    return ''.join(out)


FUNCTIONS['ASC'] = wrap_ufunc(xasc, **_kw0)


def xbahttext(value):
    from bahttext import bahttext
    return bahttext(value)


FUNCTIONS['BAHTTEXT'] = wrap_ufunc(xbahttext)

_CLEAN_TRANS = {cp: None for cp in range(0, 32)}


def xclean(value):
    return _str(value).translate(_CLEAN_TRANS)


FUNCTIONS['CLEAN'] = wrap_ufunc(xclean, **_kw0)

_re_hex = regex.compile("_x([0-9A-Z]{4})_")


def xchar(number):
    return codes.get(int(number), Error.errors['#VALUE!'])


FUNCTIONS['CHAR'] = wrap_ufunc(xchar)


def xunichar(number):
    number = int(number)
    if number:
        return chr(number)
    return Error.errors['#VALUE!']


FUNCTIONS['UNICHAR'] = FUNCTIONS['_XLFN.UNICHAR'] = wrap_ufunc(xunichar)


def xcode(character):
    if isinstance(character, sh.Token):
        raise ValueError
    return inverse_codes.get(str(character)[0], None)


FUNCTIONS["CODE"] = wrap_ufunc(
    xcode, args_parser=lambda *a: a, input_parser=lambda *a: a
)


def xunicode(character):
    if isinstance(character, sh.Token):
        raise ValueError
    return ord(str(character)[0])


FUNCTIONS["UNICODE"] = FUNCTIONS["_XLFN.UNICODE"] = wrap_ufunc(
    xunicode, args_parser=lambda *a: a, input_parser=lambda *a: a
)


def _str(text):
    if isinstance(text, bool):
        return str(text).upper()
    if isinstance(text, float) and text.is_integer():
        return '%d' % text
    return str(text)


def xexact(text1, text2):
    return _str(text1) == _str(text2)


FUNCTIONS['EXACT'] = wrap_ufunc(xexact, **_kw0)


def xfind(find_text, within_text, start_num=1):
    i = int(start_num or 0) - 1
    res = i >= 0 and _str(within_text).find(_str(find_text), i) + 1 or 0
    return res or Error.errors['#VALUE!']


FUNCTIONS['FIND'] = FUNCTIONS['FINDB'] = wrap_ufunc(xfind, **_kw0)


def xleft(from_str, num_chars=1):
    i = int(num_chars or 0)
    if i >= 0:
        return _str(from_str)[:i]
    return Error.errors['#VALUE!']


FUNCTIONS['LEFT'] = FUNCTIONS['LEFTB'] = wrap_ufunc(xleft, **_kw0)

_kw1 = {
    'input_parser': lambda text: [_str(text)],
    'args_parser': lambda *a: map(functools.partial(replace_empty, empty=''), a)
}
FUNCTIONS['LEN'] = FUNCTIONS['LENB'] = wrap_ufunc(str.__len__, **_kw1)
FUNCTIONS['LOWER'] = wrap_ufunc(str.lower, **_kw1)


def xmid(from_str, start_num, num_chars):
    i = j = int(start_num or 0) - 1
    j += int(num_chars or 0)
    if 0 <= i <= j:
        return _str(from_str)[i:j]
    return Error.errors['#VALUE!']


FUNCTIONS['MID'] = FUNCTIONS['MIDB'] = wrap_ufunc(xmid, **_kw0)


def xnumbervalue(text, decimal_sep=None, group_sep=None):
    text = _str(replace_empty(text, '0'))
    g = ',' if group_sep is None else replace_empty(group_sep)[0]
    d = '.' if decimal_sep is None else replace_empty(decimal_sep)[0]
    if g == d and group_sep is not None and decimal_sep is not None:
        return Error.errors['#VALUE!']
    g_esc = re.escape(g)
    d_esc = re.escape(d)
    number = rf'\s*[\d{g_esc}]+(?:{d_esc}[\s\d]*)?(?:\s*[eE]\s*[+-]\s*\d[\s\d]*)?[\s%]*'
    if {'(', ')'}.intersection({g, d}):
        m = re.match(f'^\s*[+-]?{number}$', text)
    else:
        m = re.match(
            f'(?:^\s*[+-]?{number}$)|(?:^\s*\({number}\)[\s%]*$)', text
        )
    if not m:
        return Error.errors['#VALUE!']
    return float(text.translate(str.maketrans({
        ' ': '', '%': '', '(': '-', ')': '', g: '', d: '.'
    }))) * 0.01 ** text.count('%')


FUNCTIONS['NUMBERVALUE'] = FUNCTIONS['_XLFN.NUMBERVALUE'] = wrap_ufunc(
    xnumbervalue, args_parser=lambda *a: a, input_parser=lambda *a: a
)
FUNCTIONS['PROPER'] = wrap_ufunc(str.title, **_kw1)


def _xregex(text, pattern, case_sensitivity=0):
    case_sensitivity = int(float(replace_empty(case_sensitivity)))
    if case_sensitivity not in (0, 1):
        raise FoundError(err=Error.errors['#VALUE!'])
    rx = re.compile(
        _str(pattern), re.IGNORECASE if case_sensitivity == 1 else 0
    )
    return rx, _str(text)


_get_first = np.vectorize(
    lambda x: x[0] if isinstance(x, list) else x, otypes=[object]
)


def _xregexextract_return_func(res, *args):
    if not res.shape:
        res = res.item()
        res = np.asarray(res, object).view(Array)
    elif res.shape == (1, 1):
        res = res[0, 0]
        res = np.asarray(res, object).view(Array)
    else:
        res = _get_first(res)
    return res


def xregexextract(text, pattern, return_mode=0, case_sensitivity=0):
    rx, text = _xregex(text, pattern, case_sensitivity)
    return_mode = int(float(replace_empty(return_mode)))
    if return_mode == 0:
        m = rx.search(text)
        r = [m.group(0) if m else None]
    elif return_mode == 1:
        r = [m.group(0) for m in rx.finditer(text)] or [None]
    elif return_mode == 2:
        m = rx.search(text)
        r = list(m.groups() or [Error.errors['#VALUE!']]) if m else [None]
    else:
        return Error.errors['#VALUE!']
    return [Error.errors['#N/A'] if v is None else v for v in r]


def xregexreplace(text, pattern, replacement, occurrence=0, case_sensitivity=0):
    rx, text = _xregex(text, pattern, case_sensitivity)
    return rx.sub(
        _str(replacement), text, count=int(float(replace_empty(occurrence)))
    )


def xregextest(text, pattern, case_sensitivity=0):
    rx, text = _xregex(text, pattern, case_sensitivity)
    return rx.match(text) is not None


FUNCTIONS['REGEXEXTRACT'] = FUNCTIONS['_XLFN.REGEXEXTRACT'] = wrap_ufunc(
    xregexextract, return_func=_xregexextract_return_func, check_nan=False,
    **_kw0
)
FUNCTIONS['REGEXREPLACE'] = FUNCTIONS['_XLFN.REGEXREPLACE'] = wrap_ufunc(
    xregexreplace, check_nan=False, **_kw0
)
FUNCTIONS['REGEXTEST'] = FUNCTIONS['_XLFN.REGEXTEST'] = wrap_ufunc(
    xregextest, check_nan=False, **_kw0
)


def xreplace(old_text, start_num, num_chars, new_text):
    old_text, new_text = _str(old_text), _str(new_text)
    i = j = int(start_num or 0) - 1
    j += int(num_chars or 0)
    if 0 <= i <= j:
        return old_text[:i] + new_text + old_text[j:]
    return Error.errors['#VALUE!']


FUNCTIONS['REPLACE'] = FUNCTIONS['REPLACEB'] = wrap_ufunc(xreplace, **_kw0)


def xrept(text, number_times):
    return _str(text) * int(float(replace_empty(number_times)))


FUNCTIONS['REPT'] = wrap_ufunc(xrept, **_kw0)


def xright(from_str, num_chars=1):
    res = xleft(_str(from_str)[::-1], num_chars)
    return res if isinstance(res, XlError) else res[::-1]


FUNCTIONS['RIGHT'] = FUNCTIONS['RIGHTB'] = wrap_ufunc(xright, **_kw0)
FUNCTIONS['TRIM'] = wrap_ufunc(str.strip, **_kw1)
FUNCTIONS['UPPER'] = wrap_ufunc(str.upper, **_kw1)


def xsearch(find_text, within_text, start_num=1):
    n = int(start_num - 1)
    n = str(within_text).lower().find(str(find_text).lower(), n)
    if n < 0:
        return Error.errors['#VALUE!']
    return n + 1


FUNCTIONS['SEARCH'] = FUNCTIONS['SEARCHB'] = wrap_ufunc(xsearch, **_kw0)


def xsubstitute(text, old_text, new_text, instance_num=None):
    text, old_text, new_text = tuple(map(_str, (text, old_text, new_text)))
    if instance_num is None:
        return text.replace(old_text, new_text)
    elif isinstance(instance_num, (
            bool, np.bool_, str, np.str_
    )) or instance_num < 1:
        return Error.errors['#VALUE!']

    parts = text.split(old_text)
    instance_num = int(instance_num)
    if instance_num > len(parts) - 1:
        return text

    return old_text.join(parts[:instance_num]) + new_text + old_text.join(
        parts[instance_num:]
    )


FUNCTIONS['SUBSTITUTE'] = wrap_ufunc(
    xsubstitute, input_parser=lambda *a: a,
    args_parser=lambda *a: (replace_empty(v, '') for v in a)
)


def xconcat(text, *args):
    it = list(flatten((text,) + args, is_not_empty))
    raise_errors(it)
    return ''.join(map(_str, it))


FUNCTIONS['_XLFN.CONCAT'] = FUNCTIONS['CONCAT'] = wrap_func(xconcat)
FUNCTIONS['_XLFN.CONCATENATE'] = FUNCTIONS['CONCATENATE'] = wrap_ufunc(
    xconcat, **_kw0
)


def xtextjoin(delimiter, ignore_empty, text, *args):
    raise_errors(delimiter, ignore_empty, text, *args)

    if ignore_empty:
        it = (flatten((text,) + args, is_not_empty))
    else:
        it = (replace_empty(v, '') for v in flatten((text,) + args, None))
    return _str(next(flatten(delimiter, None))).join(map(_str, it))


FUNCTIONS['_XLFN.TEXTJOIN'] = FUNCTIONS['TEXTJOIN'] = wrap_func(xtextjoin)
_re_format_code = regex.compile(
    r'(?P<text>"[^"]*")|'
    r'(?P<escape>\\.)|'
    r'(?P<percentage>\%)|'
    r'(?P<thousand>(?<=[#0]),(?=[#0]))|'
    r'(?P<time>[Aa][mM]/[Pp][mM]|[Aa]/[Pp]|[hH]+|[sS]+)|'
    r'(?P<date>[Yy]+|[dD]+)|'
    r'(?P<months_minutes>[mM]+)|'
    r'(?P<exp>E[+-]?)|'
    r'(?P<wrong>[e"])|'
    r'(?P<number>[#0])|'
    r'(?P<decimal>\.)|'
    r'(?P<skip>(?<=[#0]),)|'
    r'(?P<condition>;?\[[^\]]+\]|;)|'
    r'(?P<extra>$)'
)
_re_sub_condition = regex.compile(r'[\[\];]+|^\s+|\s+$')


@functools.lru_cache()
def _parse_format_code(format_code):
    formats = []
    codes = []
    types = {}
    code = []
    str_index = 0
    minutes = None
    suspended = []
    for match in _re_format_code.finditer(format_code):
        # noinspection PyUnresolvedReferences
        span = match.span()
        if str_index != span[0]:
            v = format_code[str_index:span[0]]
            sh.get_nested_dicts(types, 'extra', default=list).append(len(codes))
            codes.append(v)
        str_index = span[1]

        # noinspection PyUnresolvedReferences
        for k, v in match.groupdict().items():
            if v:
                if k == 'decimal' and k in types:
                    k = 'extra'
                elif k == 'number' and 'exp' in types:
                    k = 'exp'
                elif k == 'number' and 'decimal' in types:
                    k = 'decimal'
                elif k == 'thousand' and (k in types or 'decimal' in types):
                    k = 'skip'
                elif k == 'exp' and k in types or k == 'wrong':
                    raise
                elif k == 'condition':
                    code.extend(codes)
                    code.append(v)
                    codes = []
                    types = {}
                    formats.append((v, codes, types))
                    break
                elif k == 'time':
                    if suspended:
                        sh.get_nested_dicts(
                            types, 's' in v.lower() and k or 'date',
                            default=list
                        ).append(suspended.pop())
                    if minutes is None or 'h' in v.lower():
                        minutes = True
                elif k == 'date':
                    if suspended:
                        sh.get_nested_dicts(types, k, default=list).append(
                            suspended.pop()
                        )
                elif k == 'months_minutes':
                    if len(v) <= 2:
                        if minutes:
                            k = 'time'
                            minutes = False
                        elif minutes is None:
                            k = 'date'
                        elif len(suspended) > 0:
                            raise
                        else:
                            suspended.append(len(codes))
                            codes.append(v)
                            break
                    else:
                        k = 'date'
                sh.get_nested_dicts(types, k, default=list).append(len(codes))
                codes.append(v)
                break

    if len(suspended) == 1:
        sh.get_nested_dicts(types, 'date', default=list).append(
            suspended.pop()
        )
    code.extend(codes)
    assert ''.join(code) == format_code
    if not formats:
        formats = [('', codes, types)]
    conditions = []
    for condition, codes, types in formats:
        condition = _re_sub_condition.sub('', condition)
        if condition:
            from .operators import LOGIC_OPERATORS
            operator = '='
            for k in LOGIC_OPERATORS:
                if condition.startswith(k) and condition != k:
                    operator, condition = k, condition[len(k):]
                    break

            check = functools.partial(
                LOGIC_OPERATORS[operator], y=_text2num(condition)
            )
        else:
            check = lambda value: True
        factor = 1
        thousand = False
        if 'number' in types:
            if 'date' in types or 'time' in types:
                raise ValueError
            decimals = len(types.get('decimal', [None])) - 1
            if 'exp' in types:
                type = 'E'
            else:
                type = 'f'
                factor = (100 ** len(types.get('percentage', ())))
            thousand = 'thousand' in types and ',' or ''
            fotmat_string = f"{thousand}.{decimals}{type}"
        else:
            fotmat_string = ''
        for i in types.get('text', []):
            codes[i] = codes[i][1:-1]
        for i in types.get('escape', []):
            codes[i] = codes[i][1]
        for i in types.get('date', []):
            codes[i] = codes[i].lower()
        for i in types.get('time', []):
            if '/' not in codes[i] or len(codes[i]) == 5:
                codes[i] = codes[i].upper()
        for i in types.get('skip', []) + types.get('thousand', []):
            codes[i] = ''
        for k in ('number', 'decimal', 'exp'):
            for i in types.get(k, []):
                if codes[i] == '#':
                    codes[i] = ''
        if thousand:
            for i in types.get('number', [])[::-3]:
                if codes[i] == '0':
                    codes[i] = '0,'
        conditions.append((check, codes, types, fotmat_string, factor))
    return conditions


def _format_datetime(value, codes, types):
    codes = codes.copy()
    from datetime import datetime
    from .date import _int2date, _n2time
    value = datetime(*(_int2date(value) + _n2time(value)))
    parts = json.loads(format(
        value, '{"yyy":"%Y","yy":"%y","mm":"%m","mmm":"%b","mmmm":"%B",'
               '"dd":"%d","ddd":"%a","dddd":"%A","HH":"%H","MM":"%M",'
               '"SS":"%S","AM/PM":"%p"}'
    ))
    for k in 'mdHSM':
        parts[k] = parts[f'{k}{k}'].strip('0')
    parts['mmmmm'] = parts['mmm'][0]
    parts['m+'] = parts['mmmm']
    parts['d+'] = parts['dddd']
    parts['y'] = parts['yy']
    parts['y+'] = parts['yyy']
    parts['H+'] = parts['HH']
    parts['S+'] = parts['SS']
    parts['A/P'] = parts['AM/PM'].replace('M', '')
    parts['a/P'] = parts['A/P'].replace('A', 'a')
    parts['A/p'] = parts['A/P'].replace('P', 'p')
    parts['a/p'] = parts['A/P'].lower()
    for k in ('date', 'time'):
        for i in types.get(k, ()):
            v = codes[i]
            try:
                v = parts[v]
            except KeyError:
                v = parts[f'{v[0]}+']
            codes[i] = v
    return codes


_re_format_number = regex.compile(
    r'(?P<sign>[\+\-])?(?P<number>\d[\d,]*)(?P<decimal>\.\d+)?'
    r'(?>(?P<exp_sign>E[\+\-])0*(?P<exp>\d+))?'
)

_re_split_number = regex.compile(r'(?=,\d)|(?<!,)(?=\d)')


def _format_number(value, codes, types, fstr, mul):
    codes = codes.copy()
    parts = _re_format_number.match(format(value * mul, fstr)).groupdict('')
    # noinspection PyTypeChecker
    parts['number'] = _re_split_number.split(parts['number'].lstrip('0'))[1:]
    it = (
        (types.get('number', ())[::-1], iter(parts['number'][::-1]), True),
        (types.get('decimal', ()), iter(parts['decimal'].rstrip('0')), False),
        (types.get('exp', ())[:0:-1], iter(parts['exp'][::-1]), True)
    )
    for index, values, reverse in it:
        for i in index[:-1]:
            codes[i] = next(values, codes[i])
        for i in index[-1:]:
            v = tuple(values)
            v = ''.join(v[::-1] if reverse else v)
            codes[i] = v or codes[i]
    for i in types.get('exp', ())[:1]:
        if codes[i] == 'E-' and parts['exp_sign'] == 'E+':
            codes[i] = 'E'
        else:
            codes[i] = parts['exp_sign']
    return [parts['sign']] + codes


def xt(value):
    value = np.asarray(value, dtype=object).ravel()[0]
    raise_errors(value)

    if isinstance(value, (np.str_, str)):
        return replace_empty(value, '')
    return ''


FUNCTIONS['T'] = wrap_func(xt)


def xtext(value, format_code):
    it = _parse_format_code(str(format_code))
    if isinstance(value, (np.bool_, bool)):
        return str(value).upper()
    try:
        value = xvalue(value)
    except (ValueError, TypeError, AssertionError):
        return value
    for check, codes, types, fstr, mul in it:
        if not check(value):
            continue
        if 'date' in types or 'time' in types:
            codes = _format_datetime(value, codes, types)
        else:
            codes = _format_number(value, codes, types, fstr, mul)
        return ''.join(codes)
    raise


FUNCTIONS['TEXT'] = wrap_ufunc(xtext, input_parser=lambda *a: a)


def xvalue(value):
    if not isinstance(value, Error) and isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            from .date import xdate, _text2datetime, xtime
            value = _text2datetime(value)
            return xdate(*value[:3]) + xtime(*value[3:])
    elif isinstance(value, (np.bool_, bool)):
        raise ValueError
    return float(value)


FUNCTIONS['VALUE'] = wrap_ufunc(xvalue, input_parser=lambda *a: a)


def _xvaluetotext(text, format_type):
    if isinstance(text, sh.Token):
        if text is sh.EMPTY:
            return ''
        return _str(text)
    r = _str(text)
    if format_type and isinstance(text, str):
        return f'"{r}"'
    return r


def xvaluetotext(text, format_type=0):
    format_type = int(float(replace_empty(format_type)))
    if format_type not in (0, 1):
        return Error.errors['#VALUE!']
    return _xvaluetotext(text, format_type)


FUNCTIONS['VALUETOTEXT'] = FUNCTIONS['_XLFN.VALUETOTEXT'] = wrap_ufunc(
    xvaluetotext, input_parser=lambda *a: a, args_parser=lambda *a: a,
    check_error=lambda a, format_type=0: get_error(format_type)
)


def xarraytotext(array, format_type=0):
    format_type = int(float(replace_empty(format_type)))
    array = np.asarray(array, object)
    if format_type == 0:
        return ', '.join(
            _xvaluetotext(v, format_type)
            for v in flatten(array, None)
        )
    elif format_type == 1:
        if len(array.shape) == 1:
            r = ', '.join(
                _xvaluetotext(v, format_type)
                for v in flatten(array, None)
            )
        elif len(array.shape) == 2:
            r = ';'.join(
                ','.join(_xvaluetotext(v, format_type) for v in row)
                for row in array
            )
        return f'{{{r}}}' if r else Error.errors['#VALUE!']
    return Error.errors['#VALUE!']


FUNCTIONS['ARRAYTOTEXT'] = FUNCTIONS['_XLFN.ARRAYTOTEXT'] = wrap_ufunc(
    xarraytotext, input_parser=lambda *a: a, args_parser=lambda *a: a,
    check_error=lambda a, format_type=0: get_error(format_type), excluded={0}
)


def _parse_bool(value):
    if isinstance(value, str):
        if value.upper() == "TRUE":
            value = True
        elif value.upper() == "FALSE":
            value = False
        else:
            raise FoundError(err=Error.errors['#VALUE!'])
    else:
        value = bool(value)
    return value


def xfixed(number, decimals=2, no_commas=False):
    number = float(replace_empty(number))
    decimals = int(float(replace_empty(decimals)))
    no_commas = _parse_bool(replace_empty(no_commas, False))

    if decimals > 127:
        return Error.errors['#VALUE!']
    fmt = f"{',' if not no_commas else ''}.{max(decimals, 0)}f"

    from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP, localcontext
    number = Decimal(number)
    prec = len(number.as_tuple().digits) + decimals
    if prec <= 0:
        r = 0
    else:
        if number >= 1e16:
            quant = Decimal(1).scaleb(number.adjusted() - (15 - 1))
            number = number.quantize(quant, rounding=ROUND_DOWN)
        q = Decimal(1).scaleb(-decimals)
        with localcontext() as ctx:
            ctx.rounding = ROUND_HALF_UP
            ctx.prec = prec
            r = number.quantize(q)

        if r.is_zero():
            r = abs(r)
    return format(r, fmt)


FUNCTIONS['FIXED'] = wrap_ufunc(
    xfixed, input_parser=lambda *a: a, args_parser=lambda *a: a
)


def xtextsplit(
        text, col_delimiter, row_delimiter=None, ignore_empty=False,
        match_mode=0, pad_with=Error.errors['#N/A']
):
    ignore_empty = _parse_bool(replace_empty(ignore_empty, False))
    match_mode = int(float(replace_empty(match_mode)))
    if match_mode == 1:
        flags = re.IGNORECASE
    elif match_mode == 0:
        flags = 0
    else:
        return Error.errors['#VALUE!']
    text = replace_empty(text, Error.errors['#VALUE!'])
    raise_errors(text)
    text = _str(text)
    col_delimiter = tuple(map(_str, flatten(col_delimiter, None, True)))
    if not col_delimiter or any(not v for v in col_delimiter):
        return Error.errors['#VALUE!']
    re_col = re.compile('|'.join(map(re.escape, col_delimiter)), flags=flags)

    if row_delimiter is None:
        rows = [text]
    else:
        row_delimiter = tuple(map(_str, flatten(row_delimiter, None, True)))
        if not row_delimiter or any(not v for v in row_delimiter):
            return Error.errors['#VALUE!']
        rows = re.split('|'.join(map(
            re.escape, row_delimiter
        )), text, flags=flags)

    rows = [re_col.split(row) for row in rows]

    if ignore_empty:
        rows = [[v for v in row if v] for row in rows]
        rows = [row for row in rows if row]

    maxcols = max(map(len, rows))
    for row in rows:
        if len(row) < maxcols:
            row.extend([pad_with] * (maxcols - len(row)))

    return rows


FUNCTIONS['TEXTSPLIT'] = FUNCTIONS['_XLFN.TEXTSPLIT'] = wrap_ufunc(
    xtextsplit, input_parser=lambda *a: a, args_parser=lambda *a: a,
    excluded={1, 2, 5}, return_func=return_2d_func, check_nan=False
)


def xtextafterbefore(
        after, text, delimiter, instance_num=1, match_mode=0, match_end=0,
        if_not_found=Error.errors['#N/A']
):
    instance_num = int(float(replace_empty(instance_num)))
    match_mode = int(float(replace_empty(match_mode)))
    match_end = int(float(replace_empty(match_end)))
    if match_mode == 1:
        flags = re.IGNORECASE
    elif match_mode == 0:
        flags = 0
    else:
        return Error.errors['#VALUE!']
    text = replace_empty(text, Error.errors['#VALUE!'])
    raise_errors(text)
    text = _str(text)
    if match_end not in (0, 1) or instance_num == 0 or instance_num > len(text):
        return Error.errors['#VALUE!']
    delimiter = tuple(map(_str, flatten(delimiter, None, True)))
    if not delimiter or any(not v for v in delimiter):
        return Error.errors['#VALUE!']
    if match_end:
        text = text + delimiter[0]
    r = list(re.finditer('|'.join(map(
        re.escape, delimiter
    )), text, flags=flags))
    try:
        m = r[instance_num - 1 if instance_num > 0 else instance_num]
        return text[m.end():] if after else text[:m.start()]
    except IndexError:
        return if_not_found


FUNCTIONS['TEXTAFTER'] = FUNCTIONS['_XLFN.TEXTAFTER'] = wrap_ufunc(
    functools.partial(xtextafterbefore, True),
    input_parser=lambda *a: a,
    args_parser=lambda *a: a,
    excluded={1, 5},
    check_nan=False
)
FUNCTIONS['TEXTBEFORE'] = FUNCTIONS['_XLFN.TEXTBEFORE'] = wrap_ufunc(
    functools.partial(xtextafterbefore, False),
    input_parser=lambda *a: a,
    args_parser=lambda *a: a,
    excluded={1, 5},
    check_nan=False
)
