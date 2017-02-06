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
    for v in flatten(args, None):
        if isinstance(v, XlError) and v is not Error.errors['#N/A']:
            raise FoundError(err=v)


def wrap_func(func):
    def wrapper(*args, **kwargs):
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
    'INT': wrap_func(int),
    'PI': lambda: math.pi,
    'SUM': wrap_func(xsum),
    'AVERAGE': wrap_func(average),
    'ARRAYROW': lambda *args: np.asarray(args, object).view(Array),
    'ARRAY': lambda *args: np.asarray(args, object).view(Array),
    'IF': wrap_func(lambda c, x=True, y=False: np.where(c, x, y)),
    'IFERROR': iferror,
    'ISERROR': iserror,
    'ISERR': iserr
})
