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


numpy_ufuncs = {item: getattr(np, item) for item in dir(np)
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


def call_function(value, function, xl_error):
    def safe_eval(f, val):
        try:
            return f(val)
        except (ValueError, TypeError):
            return np.nan

    if isinstance(value, (list, tuple, np.ndarray)):
        # Make an array for the error
        err_arr = np.where(np.ones(value.shape), xl_error, None)
        result = np.reshape([safe_eval(function, item) for item in np.ravel(value)],
                            value.shape)
        # Replace `nan` with appropriate error
        return np.where(np.isnan(result), err_arr, result)
    else:
        try:
            result = function(value)
            return xl_error if result is np.nan else result
        except (TypeError, ValueError):
            return xl_error


def wrap_func(func, xl_error='#VALUE!'):
    if isinstance(xl_error, str):
        xl_error = Error.errors[xl_error]

    def wrapper(*args, **kwargs):
        # noinspection PyBroadException
        try:
            raise_errors(*args)
            if isinstance(func, str):
                return functools.partial(call_function,
                                         function=numpy_ufuncs[func],
                                         xl_error=xl_error)(*args, **kwargs)
            return func(*args, **kwargs)
        except FoundError as ex:
            return np.asarray([[ex.err]], object)
        except:
            return np.asarray([[xl_error]], object)
    return functools.update_wrapper(wrapper, func)


FUNCTIONS = collections.defaultdict(lambda: not_implemented)
FUNCTIONS.update({
    'ACOS': wrap_func('arccos', '#NUM!'),
    'ACOSH': wrap_func('arccosh', '#NUM!'),
    'ARRAY': lambda *args: np.asarray(args, object).view(Array),
    'ARRAYROW': lambda *args: np.asarray(args, object).view(Array),
    'ASIN': wrap_func('arcsin', '#NUM!'),
    'ASINH': wrap_func('arcsinh', '#NUM!'),
    'ATAN': wrap_func('arctan', '#NUM!'),
    'ATAN2': wrap_func('arctan2', '#NUM!'),
    'ATANH': wrap_func('arctanh', '#NUM!'),
    'AVERAGE': wrap_func(average),
    'COS': wrap_func('cos', '#NUM!'),
    'COSH': wrap_func('cosh', '#NUM!'),
    'EXP': wrap_func('exp', '#NUM!'),
    'IF': wrap_func(lambda c, x=True, y=False: np.where(c, x, y)),
    'IFERROR': iferror,
    'INT': wrap_func(int),
    'ISERR': iserr,
    'ISERROR': iserror,
    'LOG': wrap_func('log10', '#NUM!'),
    'LN': wrap_func('log', '#NUM!'),
    'PI': lambda: math.pi,
    'SIN': wrap_func('sin', '#NUM!'),
    'SINH': wrap_func('cosh', '#NUM!'),
    'SUM': wrap_func(xsum),
    'TAN': wrap_func('tan', '#NUM!'),
    'TANH': wrap_func('tanh', '#NUM!'),
})
