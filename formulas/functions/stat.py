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
    XlError, wrap_ufunc, replace_empty, get_error, is_not_empty, _convert_args,
    convert_nan
)

FUNCTIONS = {}


def _convert(v):
    if isinstance(v, str):
        return 0
    if isinstance(v, bool):
        return int(v)
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


def xcorrel(arr1, arr2):
    arr1, arr2 = tuple(flatten(arr1, None)), tuple(flatten(arr2, None))
    if (sh.EMPTY,) == arr1 or (sh.EMPTY,) == arr2:
        return Error.errors['#VALUE!']
    if len(arr1) != len(arr2):
        return Error.errors['#N/A']
    a12 = tuple(_forecast_known_filter(arr1, arr2))
    if len(a12) <= 1:
        return Error.errors['#DIV/0!']
    return np.corrcoef(*zip(*a12))[0, 1]


FUNCTIONS['CORREL'] = wrap_func(xcorrel)
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
FUNCTIONS['MEDIAN'] = wrap_func(functools.partial(
    xfunc, func=lambda x: convert_nan(np.median(x)), default=None
))
FUNCTIONS['MIN'] = wrap_func(functools.partial(xfunc, func=min))
FUNCTIONS['MINA'] = wrap_func(functools.partial(
    xfunc, convert=_convert, check=is_not_empty, func=min
))


def _forecast_known_filter(known_y, known_x):
    for v in zip(known_y, known_x):
        if not any(isinstance(i, (str, bool)) for i in v):
            yield v


def _args_parser_forecast(x, yp, xp):
    yp, xp = tuple(flatten(yp, check=None)), tuple(flatten(xp, check=None))
    x = replace_empty(x)
    if (sh.EMPTY,) == yp or (sh.EMPTY,) == xp:
        return x, Error.errors['#VALUE!']
    if len(yp) != len(xp):
        return x, Error.errors['#N/A']
    error = get_error(*zip(yp, xp))
    if error:
        return x, error
    yxp = tuple(_forecast_known_filter(yp, xp))
    if len(yxp) <= 1:
        return x, Error.errors['#DIV/0!']
    yp, xp = tuple(map(np.array, zip(*yxp)))
    ym, xm = yp.mean(), xp.mean()
    dx = xp - xm
    b = (dx ** 2).sum()
    if not b:
        return x, Error.errors['#DIV/0!']
    b = (dx * (yp - ym)).sum() / b
    a = ym - xm * b
    return x, a, b


def xforecast(x, a=None, b=None):
    return a + b * x


FUNCTIONS['_XLFN.FORECAST.LINEAR'] = FUNCTIONS['FORECAST'] = wrap_ufunc(
    xforecast, args_parser=_args_parser_forecast, excluded={1, 2},
    input_parser=lambda x, a, b: (_convert_args(x), a, b)
)
FUNCTIONS['FORECAST.LINEAR'] = FUNCTIONS['FORECAST']


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
