#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2025 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
Python equivalents of information Excel functions.
"""
import functools
import numpy as np
import schedula as sh
from . import (
    wrap_ranges_func, Error, Array, XlError, wrap_func, is_number, flatten,
    _text2num
)

FUNCTIONS = {}


class FalseArray(Array):
    _default = False


class TrueArray(Array):
    _default = True


def iserr(val):
    try:
        b = np.asarray([isinstance(v, XlError) and v is not Error.errors['#N/A']
                        for v in val.ravel().tolist()], bool)
        b.resize(val.shape)
        return b.view(FalseArray)
    except AttributeError:  # val is not an array.
        return iserr(np.asarray([[val]], object))[0][0].view(FalseArray)


FUNCTIONS['ISERR'] = wrap_ranges_func(iserr)


def iserror(val, check=lambda x: isinstance(x, XlError), array=TrueArray):
    try:
        b = np.asarray([check(v) for v in val.ravel().tolist()], bool)
        b.resize(val.shape)
        return b.view(array)
    except AttributeError:  # val is not an array.
        return iserror(
            np.asarray([[val]], object), check, array
        )[0][0].view(array)


def isna(value):
    return value == Error.errors['#N/A']


def xiseven_odd(number, odd=False):
    number = tuple(flatten(number, None))
    if len(number) > 1 or isinstance(number[0], bool):
        return Error.errors['#VALUE!']
    number = number[0]
    if isinstance(number, XlError):
        return number
    if number is sh.EMPTY:
        number = 0
    v = int(_text2num(number)) % 2
    return v != 0 if odd else v == 0


FUNCTIONS['ISODD'] = wrap_ranges_func(functools.partial(xiseven_odd, odd=True))
FUNCTIONS['ISEVEN'] = wrap_ranges_func(xiseven_odd)
FUNCTIONS['ISERROR'] = wrap_ranges_func(iserror)
FUNCTIONS['ISNUMBER'] = wrap_ranges_func(functools.partial(
    iserror, check=lambda x: is_number(x, xl_return=False), array=FalseArray
))
FUNCTIONS['ISBLANK'] = wrap_ranges_func(functools.partial(
    iserror, check=lambda x: x is sh.EMPTY, array=FalseArray
))
FUNCTIONS['ISTEXT'] = wrap_ranges_func(functools.partial(
    iserror, check=lambda x: isinstance(x, str) and not isinstance(x, sh.Token),
    array=FalseArray
))
FUNCTIONS['ISNONTEXT'] = wrap_ranges_func(functools.partial(
    iserror, check=lambda x: not isinstance(x, str) or isinstance(x, sh.Token),
    array=TrueArray
))
FUNCTIONS['ISLOGICAL'] = wrap_ranges_func(functools.partial(
    iserror, check=lambda x: isinstance(x, bool), array=FalseArray
))
FUNCTIONS['ISNA'] = wrap_ranges_func(functools.partial(
    iserror, check=isna, array=TrueArray
))


def xna():
    return Error.errors['#N/A']


FUNCTIONS['NA'] = wrap_func(xna)
