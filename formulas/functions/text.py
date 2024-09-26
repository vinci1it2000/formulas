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
import json
import regex
import functools
import numpy as np
import schedula as sh
from . import (
    wrap_ufunc, Error, replace_empty, XlError, flatten, wrap_func, is_not_empty,
    raise_errors, _text2num
)

FUNCTIONS = {}

codes = {
    9: "\t",
    10: "\n",
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


class HexValue(str):
    pass


_re_hex = regex.compile("^_x([0-9A-Z]{4})_$")


def xchar(number):
    number = int(number)
    if 0 < number <= 255:
        if number in codes:
            return codes[number]
        return HexValue(f'_x{hex(number)[2:].upper():0>4}_')
    return Error.errors['#VALUE!']


FUNCTIONS['CHAR'] = wrap_ufunc(xchar)


def xcode(character) -> int:
    """
    Returns the  code of a character based on codes dictionary. Input must be a singular character
    """

    if isinstance(character, HexValue):
        try:
            number = int(f'0{character[1:-1]}', 16)
            if 0 < number <= 255:
                return number
        except ValueError:
            pass
    elif isinstance(character, sh.Token):
        raise ValueError
    return inverse_codes.get(str(character)[0], None)


FUNCTIONS["CODE"] = wrap_ufunc(
    xcode, args_parser=lambda *a: a, input_parser=lambda *a: a
)


def _str(text):
    if isinstance(text, bool):
        return str(text).upper()
    if isinstance(text, float) and text.is_integer():
        return '%d' % text
    return str(text)


def xfind(find_text, within_text, start_num=1):
    i = int(start_num or 0) - 1
    res = i >= 0 and _str(within_text).find(_str(find_text), i) + 1 or 0
    return res or Error.errors['#VALUE!']


_kw0 = {
    'input_parser': lambda *a: a,
    'args_parser': lambda *a: map(functools.partial(replace_empty, empty=''), a)
}
FUNCTIONS['FIND'] = wrap_ufunc(xfind, **_kw0)


def xleft(from_str, num_chars=1):
    i = int(num_chars or 0)
    if i >= 0:
        return _str(from_str)[:i]
    return Error.errors['#VALUE!']


FUNCTIONS['LEFT'] = wrap_ufunc(xleft, **_kw0)

_kw1 = {
    'input_parser': lambda text: [_str(text)],
    'args_parser': lambda *a: map(functools.partial(replace_empty, empty=''), a)
}
FUNCTIONS['LEN'] = wrap_ufunc(str.__len__, **_kw1)
FUNCTIONS['LOWER'] = wrap_ufunc(str.lower, **_kw1)


def xmid(from_str, start_num, num_chars):
    i = j = int(start_num or 0) - 1
    j += int(num_chars or 0)
    if 0 <= i <= j:
        return _str(from_str)[i:j]
    return Error.errors['#VALUE!']


FUNCTIONS['MID'] = wrap_ufunc(xmid, **_kw0)


def xreplace(old_text, start_num, num_chars, new_text):
    old_text, new_text = _str(old_text), _str(new_text)
    i = j = int(start_num or 0) - 1
    j += int(num_chars or 0)
    if 0 <= i <= j:
        return old_text[:i] + new_text + old_text[j:]
    return Error.errors['#VALUE!']


FUNCTIONS['REPLACE'] = wrap_ufunc(xreplace, **_kw0)


def xright(from_str, num_chars=1):
    res = xleft(_str(from_str)[::-1], num_chars)
    return res if isinstance(res, XlError) else res[::-1]


FUNCTIONS['RIGHT'] = wrap_ufunc(xright, **_kw0)
FUNCTIONS['TRIM'] = wrap_ufunc(str.strip, **_kw1)
FUNCTIONS['UPPER'] = wrap_ufunc(str.upper, **_kw1)


def xsearch(find_text, within_text, start_num=1):
    n = int(start_num - 1)
    n = str(within_text).lower().find(str(find_text).lower(), n)
    if n < 0:
        return Error.errors['#VALUE!']
    return n + 1


FUNCTIONS['SEARCH'] = wrap_ufunc(xsearch, **_kw0)


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
        return value
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
