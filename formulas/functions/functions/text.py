#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
Python equivalents of text excel functions.
"""
import functools
from . import wrap_ufunc, Error, replace_empty, XlError, Array

FUNCTIONS = {}


def _str(text):
    if isinstance(text, bool):
        return str(text).upper()
    return str(text)


def xfind(find_text, within_text, start_num=1):
    i = int(start_num or 0) - 1
    res = i >= 0 and _str(within_text).find(_str(find_text), i) + 1 or 0
    return res or Error.errors['#VALUE!']


_kw0 = dict(
    input_parser=lambda *a: a,
    args_parser=lambda *a: map(functools.partial(replace_empty, empty=''), a)
)
FUNCTIONS['FIND'] = wrap_ufunc(xfind, **_kw0)


def xleft(from_str, num_chars):
    i = int(num_chars or 0)
    if i >= 0:
        return _str(from_str)[:i]
    return Error.errors['#VALUE!']


FUNCTIONS['LEFT'] = wrap_ufunc(xleft, **_kw0)


class TrimArray(Array):
    def collapse(self, shape):
        if tuple(shape) == (1, 1) != self.shape:
            return Error.errors['#VALUE!']
        return super(TrimArray, self).collapse(shape)


_kw1 = dict(
    input_parser=lambda text: [_str(text)], otype=lambda *a: TrimArray,
    args_parser=lambda *a: map(functools.partial(replace_empty, empty=''), a),
)
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


def xright(from_str, num_chars):
    res = xleft(_str(from_str)[::-1], num_chars)
    return res if isinstance(res, XlError) else res[::-1]


FUNCTIONS['RIGHT'] = wrap_ufunc(xright, **_kw0)
FUNCTIONS['TRIM'] = wrap_ufunc(str.strip, **_kw1)
FUNCTIONS['UPPER'] = wrap_ufunc(str.upper, **_kw1)
