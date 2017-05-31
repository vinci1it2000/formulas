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
from ..errors import FunctionError, FoundError
from ..tokens.operand import XlError, Error


inf = float('inf')
ufuncs = {item: getattr(np, item) for item in dir(np)
          if isinstance(getattr(np, item), np.ufunc)}


def is_number(number):
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


def xsum(*args):
    return sum(list(flatten(args)))


def xmax(*args):
    return max([arg for arg in flatten(args) if is_number(arg)])


def xmin(*args):
    return min([arg for arg in flatten(args) if is_number(arg)])


def average(*args):
    l = list(flatten(args))
    return sum(l) / len(l)


# noinspection PyUnusedLocal
def not_implemented(*args, **kwargs):
    raise FunctionError()


class Array(np.ndarray):
    pass


def iserr(val):
    b = np.asarray([isinstance(v, XlError) and v is not Error.errors['#N/A']
                    for v in val.ravel().tolist()], bool)
    b.resize(val.shape)
    return b


def iserror(val):
    b = np.asarray([isinstance(v, XlError) for v in val.ravel().tolist()], bool)
    b.resize(val.shape)
    return b


def iferror(val, val_if_error):
    return np.where(iserror(val), val_if_error, val)


def raise_errors(*args):
    # noinspection PyTypeChecker
    for v in flatten(args, None):
        if isinstance(v, XlError) and v is not Error.errors['#N/A']:
            raise FoundError(err=v)


def call_ufunc(value, ufunc):
    """Helps call a numpy universal function (ufunc)."""

    def safe_eval(f, val):
        try:
            res = f(val)
            return res if all((res < inf, res > -inf)) else np.nan
        except (ValueError, TypeError):
            return Error.errors['#VALUE!']

    if isinstance(value, (list, tuple, np.ndarray)):
        # Make an array for the error
        err_arr = np.where(np.ones(value.shape), Error.errors['#NUM!'], None)
        result = np.reshape([safe_eval(ufunc, item) for item in np.ravel(value)],
                            value.shape)
        # Replace `nan` with appropriate error
        return np.where(np.isnan(result), err_arr, result)
    else:
        result = safe_eval(ufunc, value)
        if result is np.nan:
            return Error.errors['#NUM!']
        else:
            return result


def wrap_func(func):
    if func in ufuncs:
        func = functools.partial(call_ufunc, ufunc=ufuncs[func])

    def wrapper(*args, **kwargs):
        # noinspection PyBroadException
        try:
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
    'ATAN2': wrap_func('arctan2'),
    'ATANH': wrap_func('arctanh'),
    'AVERAGE': wrap_func(average),
    'COS': wrap_func('cos'),
    'COSH': wrap_func('cosh'),
    'EXP': wrap_func('exp'),
    'IF': wrap_func(lambda c, x=True, y=False: np.where(c, x, y)),
    'IFERROR': iferror,
    'INT': wrap_func(int),
    'ISERR': iserr,
    'ISERROR': iserror,
    'LOG': wrap_func('log10'),
    'LN': wrap_func('log'),
    'MAX': wrap_func(xmax),
    'MIN': wrap_func(xmin),
    'PI': lambda: math.pi,
    'SIN': wrap_func('sin'),
    'SINH': wrap_func('sinh'),
    'SUM': wrap_func(xsum),
    'TAN': wrap_func('tan'),
    'TANH': wrap_func('tanh'),
})
