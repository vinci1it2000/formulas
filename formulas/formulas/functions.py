#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
Python equivalents of various excel functions.
"""
import math
import functools
import collections
import numpy as np
from . import replace_empty, not_implemented, Array, wrap_func
from ..errors import FoundError
from ..tokens.operand import XlError, Error

FUNCTIONS = collections.defaultdict(lambda: not_implemented)


def get_error(*vals):
    for v in flatten(vals, None):
        if isinstance(v, XlError):
            return v


def raise_errors(*args):
    # noinspection PyTypeChecker
    v = get_error(*args)
    if v:
        raise FoundError(err=v)


def is_number(number):
    if isinstance(number, bool):
        return False
    elif not isinstance(number, Error):
        try:
            float(number)
        except (ValueError, TypeError):
            return False
    return True


def flatten(l, check=is_number):
    if isinstance(l, collections.Iterable) and not isinstance(l, str):
        for el in l:
            yield from flatten(el, check)
    elif not check or check(l):
        yield l


def wrap_ufunc(
        func, input_parser=lambda *a: map(float, a), check_error=get_error,
        args_parser=lambda *a: map(replace_empty, a)):
    """Helps call a numpy universal function (ufunc)."""

    def safe_eval(*vals):
        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                r = check_error(*vals) or func(*input_parser(*vals))
            if not isinstance(r, (XlError, str)):
                r = (np.isnan(r) or np.isinf(r)) and Error.errors['#NUM!'] or r
        except (ValueError, TypeError):
            r = Error.errors['#VALUE!']
        return r

    # noinspection PyUnusedLocal
    def wrapper(*args, **kwargs):
        args = args_parser(*args)
        return np.vectorize(safe_eval, otypes=[object])(*args).view(Array)

    return wrap_func(functools.update_wrapper(wrapper, func))


FUNCTIONS['ABS'] = wrap_ufunc(np.abs)
FUNCTIONS['ACOS'] = wrap_ufunc(np.arccos)
FUNCTIONS['ACOSH'] = wrap_ufunc(np.arccosh)
FUNCTIONS['_XLFN.ACOT'] = FUNCTIONS['ACOT'] = wrap_ufunc(
    lambda x: (np.arctan(np.divide(1, x)) + np.pi) % np.pi
)
FUNCTIONS['ACOTH'] = wrap_ufunc(lambda x: np.arctanh(np.divide(1, x)))
FUNCTIONS['_XLFN.ACOTH'] = FUNCTIONS['ACOTH']


def xarabic(text):
    num = (1000, 500, 100, 50, 10, 5, 1)
    it = map(num.__getitem__, map('MDCLXVI'.index, str(text).upper()[::-1]))
    res, add, p = 0, True, -1
    for v in it:
        if p != v:
            add = p < v
        p = v
        if add:
            res += v
        else:
            res -= v
    return res


FUNCTIONS['_XLFN.ARABIC'] = FUNCTIONS['ARABIC'] = wrap_ufunc(
    xarabic, input_parser=lambda *a: a,
    args_parser=lambda *a: (replace_empty(x, '') for x in a)
)

FUNCTIONS['ARRAY'] = lambda *args: np.asarray(args, object).view(Array)
FUNCTIONS['ARRAYROW'] = lambda *args: np.asarray(args, object).view(Array)
FUNCTIONS['ASIN'] = wrap_ufunc(np.arcsin)
FUNCTIONS['ASINH'] = wrap_ufunc(np.arcsinh)
FUNCTIONS['ATAN'] = wrap_ufunc(np.arctan)


def xarctan2(x, y):
    return x == y == 0 and Error.errors['#DIV/0!'] or np.arctan2(y, x)


FUNCTIONS['ATAN2'] = wrap_ufunc(xarctan2)
FUNCTIONS['ATANH'] = wrap_ufunc(np.arctanh)


def average(*args):
    l = list(flatten(args))
    return sum(l) / len(l)

def irr(*args):
    l = list(flatten(args))
    return np.irr(l)

def left(from_str, num_chars):
    return str(from_str)[:num_chars]


def mid(from_str, start_num, num_chars):
    return str(from_str)[(start_num-1):((start_num-1)+num_chars)]


def right(from_str, num_chars):
    out = str(from_str)[(0 - int(num_chars)):]
    return str(out)


def find(find_text, within_text, *args):
    if len(args) > 0:
        start_num = (args[0] - 1)
    else:
        start_num = 0
    return str(within_text).find(str(find_text), start_num)


def trim(text):
    return text.strip()


def replace(old_text, start_num, num_chars, new_text):
    return old_text[:(start_num - 1)] + new_text + old_text[(start_num - 1)+num_chars:]


FUNCTIONS['AVERAGE'] = wrap_func(average)
FUNCTIONS['COS'] = wrap_ufunc(np.cos)
FUNCTIONS['COSH'] = wrap_ufunc(np.cosh)


def xcot(x, func=np.tan):
    x = func(x)
    return (1 / x) if x else Error.errors['#DIV/0!']


FUNCTIONS['COT'] = FUNCTIONS['_XLFN.COT'] = wrap_ufunc(xcot)
FUNCTIONS['COTH'] = FUNCTIONS['_XLFN.COTH'] = wrap_ufunc(
    functools.partial(xcot, func=np.tanh)
)
FUNCTIONS['CSC'] = FUNCTIONS['_XLFN.CSC'] = wrap_ufunc(
    functools.partial(xcot, func=np.sin)
)
FUNCTIONS['CSCH'] = FUNCTIONS['_XLFN.CSCH'] = wrap_ufunc(
    functools.partial(xcot, func=np.sinh)
)


def xceiling(num, sig, ceil=math.ceil, dfl=0):
    if sig == 0:
        return dfl
    elif sig < 0 < num:
        return np.nan
    return ceil(num / sig) * sig


FUNCTIONS['CEILING'] = wrap_ufunc(xceiling)


def xceiling_math(num, sig=None, mode=0, ceil=math.ceil):
    if sig == 0:
        return 0
    elif sig is None:
        x, sig = abs(num), 1
    else:
        sig = abs(sig)
        x = num / sig
    if mode and num < 0:
        return -ceil(abs(x)) * sig
    return ceil(x) * sig


FUNCTIONS['CEILING.MATH'] = wrap_ufunc(xceiling_math)
FUNCTIONS['_XLFN.CEILING.MATH'] = FUNCTIONS['CEILING.MATH']
FUNCTIONS['CEILING.PRECISE'] = FUNCTIONS['CEILING.MATH']
FUNCTIONS['_XLFN.CEILING.PRECISE'] = FUNCTIONS['CEILING.PRECISE']
FUNCTIONS['DEGREES'] = wrap_ufunc(np.degrees)
FUNCTIONS['EXP'] = wrap_ufunc(np.exp)
FUNCTIONS['FLOOR'] = wrap_ufunc(
    functools.partial(xceiling, ceil=math.floor, dfl=Error.errors['#DIV/0!'])
)
FUNCTIONS['_XLFN.FLOOR.MATH'] = FUNCTIONS['FLOOR.MATH'] = wrap_ufunc(
    functools.partial(xceiling_math, ceil=math.floor)
)
FUNCTIONS['FLOOR.PRECISE'] = FUNCTIONS['FLOOR.MATH']
FUNCTIONS['_XLFN.FLOOR.PRECISE'] = FUNCTIONS['FLOOR.MATH']
FUNCTIONS['IF'] = wrap_func(lambda c, x=True, y=False: np.where(c, x, y))


def iferror(val, val_if_error):
    return np.where(iserror(val), val_if_error, val)


FUNCTIONS['IFERROR'] = iferror
FUNCTIONS['INT'] = wrap_ufunc(int)


def iserr(val):
    try:
        b = np.asarray([isinstance(v, XlError) and v is not Error.errors['#N/A']
                        for v in val.ravel().tolist()], bool)
        b.resize(val.shape)
        return b
    except AttributeError:  # val is not an array.
        return iserr(np.asarray([[val]], object))[0][0]


FUNCTIONS['ISERR'] = iserr


def iserror(val):
    try:
        b = np.asarray([isinstance(v, XlError)
                        for v in val.ravel().tolist()], bool)
        b.resize(val.shape)
        return b
    except AttributeError:  # val is not an array.
        return iserror(np.asarray([[val]], object))[0][0]


FUNCTIONS['ISERROR'] = iserror
FUNCTIONS['LOG10'] = wrap_ufunc(np.log10)
FUNCTIONS['LOG'] = wrap_ufunc(
    lambda x, base=10: np.log(x) / np.log(base) if base else np.nan
)
FUNCTIONS['LN'] = wrap_ufunc(np.log)


def xmax(*args):
    raise_errors(args)
    return max(list(flatten(args)) or [0])


FUNCTIONS['MAX'] = wrap_func(xmax)


def xmin(*args):
    raise_errors(args)
    return min(list(flatten(args)) or [0])


FUNCTIONS['MIN'] = wrap_func(xmin)


def xmod(x, y):
    return y == 0 and Error.errors['#DIV/0!'] or np.mod(x, y)


FUNCTIONS['MOD'] = wrap_ufunc(xmod)
FUNCTIONS['PI'] = lambda: math.pi


def xpower(number, power):
    if number == 0:
        if power == 0:
            return Error.errors['#NUM!']
        if power < 0:
            return Error.errors['#DIV/0!']
    return np.power(number, power)


FUNCTIONS['POWER'] = wrap_ufunc(xpower)
FUNCTIONS['RADIANS'] = wrap_ufunc(np.radians)


def _xroman(form):
    form = int(form + 1)
    num, let = (1000, 500, 100, 50, 10, 5, 1), 'MDCLXVI'
    for i, (n, l) in enumerate(zip(num, let), 1):
        yield n, l
        y = []
        for v, k in zip(num[i:], let[i:]):
            v = n - v
            if v not in num:
                y.append((v, k + l))
                if len(y) == form:
                    break
        yield from y[::-1]


def xroman(num, form=0):
    num, form = int(num), not isinstance(form, bool) and int(form or 0) or 0
    if not (0 <= num < 4000 and 0 <= form <= 4):
        raise ValueError()

    result = ""
    for i, n in _xroman(form):
        if not num:
            break
        count = int(num / i)
        result += n * count
        num -= i * count
    return result


FUNCTIONS['ROMAN'] = wrap_ufunc(xroman, input_parser=lambda *a: a)
FUNCTIONS['SIN'] = wrap_ufunc(np.sin)
FUNCTIONS['SINH'] = wrap_ufunc(np.sinh)


def xsumproduct(*args):
    # Check all arrays are the same length
    # Excel returns #VAlUE! error if they don't match
    raise_errors(args)
    assert len(set(arg.size for arg in args)) == 1
    inputs = []
    for a in args:
        a = a.ravel()
        x = np.zeros_like(a, float)
        b = np.vectorize(is_number)(a)
        x[b] = a[b]
        inputs.append(x)

    return np.sum(np.prod(inputs, axis=0))


FUNCTIONS['SUMPRODUCT'] = wrap_func(xsumproduct)
FUNCTIONS['SQRT'] = wrap_ufunc(np.sqrt)


def xsum(*args):
    raise_errors(args)
    return sum(list(flatten(args)))


FUNCTIONS['SUM'] = wrap_func(xsum)
FUNCTIONS['TAN'] = wrap_ufunc(np.tan)
FUNCTIONS['TANH'] = wrap_ufunc(np.tanh)
FUNCTIONS.update({
    'LEFT': wrap_func(left),
    'MID': wrap_func(mid),
    'RIGHT': wrap_func(right),
    'FIND': wrap_func(find),
    'TRIM': wrap_func(trim),
    'LEN': lambda x: len(str(x)),
    'REPLACE': wrap_func(replace),
    'UPPER': lambda x: str(x).upper(),
    'LOWER': lambda x: str(x).lower()
})
