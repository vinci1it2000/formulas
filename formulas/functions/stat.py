#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2021 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
Python equivalents of statistical Excel functions.
"""
import math
import functools
import numpy as np
import schedula as sh
from . import (
    raise_errors, flatten, wrap_func, Error, is_number, _text2num, xfilter,
    XlError, wrap_ufunc, replace_empty, get_error, is_not_empty
)

FUNCTIONS = {}


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
    xfunc, check=is_not_empty, func=len, _raise=False, default=None
))
FUNCTIONS['COUNTBLANK'] = wrap_func(functools.partial(
    xfunc, check=lambda x: (x == '' or x is sh.EMPTY), func=len,
    _raise=False, default=None
))
FUNCTIONS['COUNTIF'] = wrap_func(functools.partial(
    xfilter, len, operating_range=None
))


def xsort(values, k, large=True):
    err = get_error(k)
    if err:
        return err
    k = float(_text2num(k))
    if isinstance(values, XlError):
        return values
    if 1 <= k <= len(values):
        if large:
            k = -k
        else:
            k -= 1
        return values[math.floor(k)]
    return Error.errors['#NUM!']


def _sort_parser(values, k):
    err = get_error(values)
    if err:
        return err, k
    values = np.array(tuple(flatten(
        values, lambda v: not isinstance(v, (str, bool))
    )), float)
    values.sort()
    return values, replace_empty(k)


FUNCTIONS['LARGE'] = wrap_ufunc(
    xsort, args_parser=_sort_parser, excluded={0}, check_error=lambda *a: None,
    input_parser=lambda *a: a
)

FUNCTIONS['SMALL'] = wrap_ufunc(
    xsort, args_parser=_sort_parser, excluded={0}, check_error=lambda *a: None,
    input_parser=lambda values, k: (values, k, False)
)
FUNCTIONS['MAX'] = wrap_func(xfunc)
FUNCTIONS['MAXA'] = wrap_func(functools.partial(
    xfunc, convert=_convert, check=is_not_empty
))
FUNCTIONS['MIN'] = wrap_func(functools.partial(xfunc, func=min))
FUNCTIONS['MINA'] = wrap_func(functools.partial(
    xfunc, convert=_convert, check=is_not_empty, func=min
))


def xstdev(args, ddof=1, func=np.std):
    if len(args) <= ddof:
        return Error.errors['#DIV/0!']
    return func(args, ddof=ddof)


FUNCTIONS['_XLFN.STDEV.S'] = FUNCTIONS['STDEV.S'] = wrap_func(functools.partial(
    xfunc, func=xstdev
))
FUNCTIONS['_XLFN.STDEV.P'] = FUNCTIONS['STDEV.P'] = wrap_func(functools.partial(
    xfunc, func=functools.partial(xstdev, ddof=0), default=None
))
FUNCTIONS['STDEVA'] = wrap_func(functools.partial(
    xfunc, convert=_convert, check=is_not_empty, func=xstdev
))
FUNCTIONS['STDEVPA'] = wrap_func(functools.partial(
    xfunc, convert=_convert, check=is_not_empty, func=functools.partial(
        xstdev, ddof=0
    ), default=None
))


FUNCTIONS['_XLFN.VAR.S'] = FUNCTIONS['VAR.S'] = wrap_func(functools.partial(
    xfunc, func=functools.partial(xstdev, func=np.var)
))
FUNCTIONS['_XLFN.VAR.P'] = FUNCTIONS['VAR.P'] = wrap_func(functools.partial(
    xfunc, func=functools.partial(xstdev, ddof=0, func=np.var), default=None
))
FUNCTIONS['VARA'] = wrap_func(functools.partial(
    xfunc, convert=_convert, check=is_not_empty, func=functools.partial(
        xstdev, func=np.var
    )
))
FUNCTIONS['VARPA'] = wrap_func(functools.partial(
    xfunc, convert=_convert, check=is_not_empty, func=functools.partial(
        xstdev, ddof=0, func=np.var
    ), default=None
))
