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
import functools
import collections
import math
import numpy as np
from . import replace_empty, not_implemented, Array
from ..errors import FoundError
from ..tokens.operand import XlError, Error

ufuncs = {item: getattr(np, item) for item in dir(np)
          if isinstance(getattr(np, item), np.ufunc)}


def xpower(number, power):
    if number == 0:
        if power == 0:
            return Error.errors['#NUM!']
        if power < 0:
            return Error.errors['#DIV/0!']
    return np.power(number, power)


ufuncs['power'] = xpower


def xarctan2(x, y):
    return x == y == 0 and Error.errors['#DIV/0!'] or np.arctan2(x, y)


ufuncs['arctan2'] = xarctan2


def xmod(x, y):
    return y == 0 and Error.errors['#DIV/0!'] or np.mod(x, y)


ufuncs['mod'] = xmod


def is_number(number):
    if not isinstance(number, Error):
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


def xsumproduct(*args):
    # Check all arrays are the same length
    # Excel returns #VAlUE! error if they don't match
    assert len(set(arg.size for arg in args)) == 1
    inputs = []
    for a in args:
        a = a.ravel()
        x = np.zeros_like(a, float)
        b = np.vectorize(is_number)(a)
        x[b] = a[b]
        inputs.append(x)

    return np.sum(np.prod(inputs, axis=0))


def xsum(*args):
    return sum(list(flatten(args)))


def xmax(*args):
    return max([arg for arg in flatten(args) if is_number(arg)])


def xmin(*args):
    return min([arg for arg in flatten(args) if is_number(arg)])


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


def iserr(val):
    try:
        b = np.asarray([isinstance(v, XlError) and v is not Error.errors['#N/A']
                        for v in val.ravel().tolist()], bool)
        b.resize(val.shape)
        return b
    except AttributeError:  # val is not an array.
        return iserr(np.asarray([[val]], object))[0][0]


def iserror(val):
    try:
        b = np.asarray([isinstance(v, XlError)
                        for v in val.ravel().tolist()], bool)
        b.resize(val.shape)
        return b
    except AttributeError:  # val is not an array.
        return iserror(np.asarray([[val]], object))[0][0]


def iferror(val, val_if_error):
    return np.where(iserror(val), val_if_error, val)


def raise_errors(*args):
    # noinspection PyTypeChecker
    for v in flatten(args, None):
        if isinstance(v, XlError):
            raise FoundError(err=v)


def call_ufunc(ufunc, *args):
    """Helps call a numpy universal function (ufunc)."""

    def safe_eval(*vals):
        try:
            r = ufunc(*map(float, vals))
            if not isinstance(r, XlError) and (np.isnan(r) or np.isinf(r)):
                r = Error.errors['#NUM!']
        except (ValueError, TypeError):
            r = Error.errors['#VALUE!']
        return r

    res = np.vectorize(safe_eval, otypes=[object])(*map(replace_empty, args))
    return res.view(Array)


def wrap_func(func, args_indices=None):
    if func in ufuncs:
        func = functools.partial(call_ufunc, ufuncs[func])

    def wrapper(*args, **kwargs):
        # noinspection PyBroadException
        try:
            args = args_indices and [args[i] for i in args_indices] or args
            raise_errors(*args)
            return func(*args, **kwargs)
        except FoundError as ex:
            return np.asarray([[ex.err]], object)
        except:
            return np.asarray([[Error.errors['#VALUE!']]], object)

    return functools.update_wrapper(wrapper, func)


FUNCTIONS = collections.defaultdict(lambda: not_implemented)
FUNCTIONS.update({
    'ABS': wrap_func('abs'),
    'ACOS': wrap_func('arccos'),
    'ACOSH': wrap_func('arccosh'),
    'ARRAY': lambda *args: np.asarray(args, object).view(Array),
    'ARRAYROW': lambda *args: np.asarray(args, object).view(Array),
    'ASIN': wrap_func('arcsin'),
    'ASINH': wrap_func('arcsinh'),
    'ATAN': wrap_func('arctan'),
    'ATAN2': wrap_func('arctan2', (1, 0)),
    'ATANH': wrap_func('arctanh'),
    'AVERAGE': wrap_func(average),
    'COS': wrap_func('cos'),
    'COSH': wrap_func('cosh'),
    'DEGREES': wrap_func('degrees'),
    'EXP': wrap_func('exp'),
    'IF': wrap_func(lambda c, x=True, y=False: np.where(c, x, y)),
    'IFERROR': iferror,
    'INT': wrap_func(int),
    'IRR': wrap_func(irr),
    'ISERR': iserr,
    'ISERROR': iserror,
    'LOG': wrap_func('log10'),
    'LN': wrap_func('log'),
    'MAX': wrap_func(xmax),
    'MIN': wrap_func(xmin),
    'MOD': wrap_func('mod'),
    'PI': lambda: math.pi,
    'POWER': wrap_func('power'),
    'RADIANS': wrap_func('radians'),
    'SIN': wrap_func('sin'),
    'SINH': wrap_func('sinh'),
    'SUMPRODUCT': wrap_func(xsumproduct),
    'SQRT': wrap_func('sqrt'),
    'SUM': wrap_func(xsum),
    'TAN': wrap_func('tan'),
    'TANH': wrap_func('tanh'),
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
