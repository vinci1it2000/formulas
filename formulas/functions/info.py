#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2022 European Commission (JRC);
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


class IsErrArray(Array):
    _default = False
    _collapse_value = True


def iserr(val):
    try:
        b = np.asarray([isinstance(v, XlError) and v is not Error.errors['#N/A']
                        for v in val.ravel().tolist()], bool)
        b.resize(val.shape)
        return b.view(IsErrArray)
    except AttributeError:  # val is not an array.
        return iserr(np.asarray([[val]], object))[0][0].view(IsErrArray)


FUNCTIONS['ISERR'] = wrap_ranges_func(iserr)


class IsErrorArray(IsErrArray):
    _default = True


def iserror(val, check=lambda x: isinstance(x, XlError), array=IsErrorArray):
    try:
        b = np.asarray([check(v) for v in val.ravel().tolist()], bool)
        b.resize(val.shape)
        return b.view(array)
    except AttributeError:  # val is not an array.
        return iserror(
            np.asarray([[val]], object), check, array
        )[0][0].view(array)


class IsNumberArray(IsErrArray):
    _collapse_value = False


class IsNaArray(IsErrArray):
    _collapse_value = False
    _default = True


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
    iserror, check=lambda x: is_number(x, xl_return=False), array=IsNumberArray
))
FUNCTIONS['ISBLANK'] = wrap_ranges_func(functools.partial(
    iserror, check=lambda x: x is sh.EMPTY, array=IsNumberArray
))
FUNCTIONS['ISTEXT'] = wrap_ranges_func(functools.partial(
    iserror, check=lambda x: isinstance(x, str) and not isinstance(x, sh.Token),
    array=IsNumberArray
))
FUNCTIONS['ISNONTEXT'] = wrap_ranges_func(functools.partial(
    iserror, check=lambda x: not isinstance(x, str) or isinstance(x, sh.Token),
    array=IsErrorArray
))
FUNCTIONS['ISLOGICAL'] = wrap_ranges_func(functools.partial(
    iserror, check=lambda x: isinstance(x, bool), array=IsNumberArray
))
FUNCTIONS['ISNA'] = wrap_ranges_func(functools.partial(
    iserror, check=isna, array=IsNaArray
))


def xna():
    return Error.errors['#N/A']


FUNCTIONS['NA'] = wrap_func(xna)
