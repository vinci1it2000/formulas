#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
Python equivalents of statistical Excel functions.
"""
import functools
import schedula as sh
from . import (
    raise_errors, flatten, wrap_func, Error, is_number, _text2num, xfilter,
    XlError
)

FUNCTIONS = {}


def is_not_empty(v):
    return v is not sh.EMPTY


def _convert(v):
    if isinstance(v, str):
        return 0
    if isinstance(v, bool):
        return int(v)
    return v


def _convert_args(v):
    if isinstance(v, XlError):
        return v
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, str):
        return float(_text2num(v))
    return v


def xfunc(*args, func=max, check=is_number, convert=None, default=0,
          _raise=True):
    _raise and raise_errors(args)
    it = flatten(map(_convert_args, args), check=check)
    default = [] if default is None else [default]
    return func(list(map(convert, it) if convert else it) or default)


def _xaverage(v):
    if v:
        return sum(v) / len(v)
    return Error.errors['#DIV/0!']


xaverage = functools.partial(xfunc, func=_xaverage, default=None)
FUNCTIONS['AVERAGE'] = wrap_func(xaverage)
FUNCTIONS['AVERAGEA'] = wrap_func(functools.partial(
    xfunc, convert=_convert, check=is_not_empty, func=_xaverage, default=None
))
FUNCTIONS['AVERAGEIF'] = wrap_func(functools.partial(xfilter, xaverage))
FUNCTIONS['COUNT'] = wrap_func(functools.partial(
    xfunc, func=len, _raise=False, default=None,
    check=lambda x: is_number(x) and not isinstance(x, XlError)
))
FUNCTIONS['COUNTA'] = wrap_func(functools.partial(
    xfunc, convert=_convert, check=is_not_empty, func=len, _raise=False,
    default=None
))
FUNCTIONS['COUNTIF'] = wrap_func(functools.partial(
    xfilter, len, operating_range=None
))

FUNCTIONS['MAX'] = wrap_func(xfunc)
FUNCTIONS['MAXA'] = wrap_func(functools.partial(
    xfunc, convert=_convert, check=is_not_empty
))
FUNCTIONS['MIN'] = wrap_func(functools.partial(xfunc, func=min))
FUNCTIONS['MINA'] = wrap_func(functools.partial(
    xfunc, convert=_convert, check=is_not_empty, func=min
))
