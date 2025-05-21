#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2025 European Commission (JRC);
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
    convert_nan, FoundError, xfilters
)
from statistics import NormalDist

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
FUNCTIONS['AVERAGEIFS'] = wrap_func(functools.partial(xfilters, xaverage))


def xcorrel(arr1, arr2):
    try:
        arr1, arr2 = _parse_yxp(arr1, arr2)
    except FoundError as ex:
        return ex.err
    return np.corrcoef(arr1, arr2)[0, 1]


FUNCTIONS['CORREL'] = wrap_func(xcorrel)
FUNCTIONS['COUNT'] = wrap_func(functools.partial(
    xfunc, func=len, _raise=False, default=None,
    check=functools.partial(is_number, xl_return=False)
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
FUNCTIONS['COUNTIFS'] = wrap_func(functools.partial(
    xfilters, len, None
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
    if isinstance(values, XlError):
        raise FoundError(err=values)
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
FUNCTIONS['_XLFN.MAXIFS'] = FUNCTIONS['MAXIFS'] = wrap_func(
    functools.partial(xfilters, xfunc)
)

FUNCTIONS['MEDIAN'] = wrap_func(functools.partial(
    xfunc, func=lambda x: convert_nan(np.median(x) if x else np.nan),
    default=None
))
FUNCTIONS['MIN'] = wrap_func(functools.partial(xfunc, func=min))
FUNCTIONS['MINA'] = wrap_func(functools.partial(
    xfunc, convert=_convert, check=is_not_empty, func=min
))
FUNCTIONS['_XLFN.MINIFS'] = FUNCTIONS['MINIFS'] = wrap_func(functools.partial(
    xfilters, functools.partial(xfunc, func=min)
))


def _forecast_known_filter(known_y, known_x):
    for v in zip(known_y, known_x):
        if not any(isinstance(i, (str, bool)) for i in v):
            yield v


def xslope(yp, xp):
    try:
        a, b = _slope_coeff(*map(np.array, _parse_yxp(yp, xp)))
    except FoundError as ex:
        return ex.err
    return b


FUNCTIONS['SLOPE'] = wrap_func(xslope)


def _parse_yxp(yp, xp):
    yp, xp = tuple(flatten(yp, check=None)), tuple(flatten(xp, check=None))
    if (sh.EMPTY,) == yp or (sh.EMPTY,) == xp:
        raise FoundError(err=Error.errors['#VALUE!'])
    if len(yp) != len(xp):
        raise FoundError(err=Error.errors['#N/A'])
    raise_errors(*zip(yp, xp))
    yxp = tuple(_forecast_known_filter(yp, xp))
    if len(yxp) <= 1:
        raise FoundError(err=Error.errors['#DIV/0!'])
    return tuple(zip(*yxp))


def _slope_coeff(yp, xp):
    ym, xm = yp.mean(), xp.mean()
    dx = xp - xm
    b = (dx ** 2).sum()
    if not b:
        raise FoundError(err=Error.errors['#DIV/0!'])
    b = (dx * (yp - ym)).sum() / b
    a = ym - xm * b
    return a, b


def _args_parser_forecast(x, yp, xp):
    x = replace_empty(x)
    try:
        a, b = _slope_coeff(*map(np.array, _parse_yxp(yp, xp)))
    except FoundError as ex:
        return x, ex.err
    return x, a, b


def xforecast(x, a=None, b=None):
    return a + b * x


FUNCTIONS['_XLFN.FORECAST.LINEAR'] = FUNCTIONS['FORECAST'] = wrap_ufunc(
    xforecast, args_parser=_args_parser_forecast, excluded={1, 2},
    input_parser=lambda x, a, b: (_convert_args(x), a, b)
)
FUNCTIONS['FORECAST.LINEAR'] = FUNCTIONS['FORECAST']


def xnormdist(z, mu, sigma, cumulative=True):
    if isinstance(cumulative, str):
        if cumulative.lower() in ('true', 'false'):
            cumulative = cumulative.lower() == 'true'
        else:
            return Error.errors['#VALUE!']
    if sigma <= 0:
        return Error.errors['#NUM!']
    norm = NormalDist(mu=mu, sigma=sigma)
    return norm.cdf(z) if cumulative else norm.pdf(z)


def xnorminv(z, mu=0, sigma=1):
    if z <= 0.0 or z >= 1.0 or sigma <= 0:
        return Error.errors['#NUM!']
    norm = NormalDist(mu=mu, sigma=sigma)
    return norm.inv_cdf(z)


FUNCTIONS['_XLFN.NORM.DIST'] = FUNCTIONS['NORM.DIST'] = wrap_ufunc(
    xnormdist,
    input_parser=lambda *a: tuple(map(_convert_args, a[:-1])) + a[-1:]
)
FUNCTIONS['_XLFN.NORM.INV'] = FUNCTIONS['NORM.INV'] = wrap_ufunc(
    xnorminv,
    input_parser=lambda *a: tuple(map(_convert_args, a))
)
FUNCTIONS['_XLFN.NORM.S.DIST'] = FUNCTIONS['NORM.S.DIST'] = wrap_ufunc(
    xnormdist,
    input_parser=lambda x, *a: (_convert_args(x), 0, 1) + a
)
FUNCTIONS['_XLFN.NORM.S.INV'] = FUNCTIONS['NORM.S.INV'] = wrap_ufunc(
    xnorminv,
    input_parser=lambda x: (_convert_args(x),)
)
_percentile_kw = {
    'excluded': {0},
    'input_parser': lambda v, q: (v, _convert_args(q)),
    'check_error': lambda *a: get_error(*a[::-1]),
    'args_parser': lambda v, q: (
        list(flatten(v, drop_empty=True)), replace_empty(q)
    )
}


def xpercentile(v, p, exclusive=False):
    if len(v) == 0 or not is_number(p) or p < 0 or p > 1:
        return Error.errors['#NUM!']
    if exclusive:
        n = len(v)
        rank = (n + 1) * p
        if rank < 1 or rank > n:
            return Error.errors['#NUM!']
    return np.percentile(v, p * 100, method=exclusive and 'weibull' or 'linear')


FUNCTIONS['_XLFN.PERCENTILE.EXC'] = FUNCTIONS['PERCENTILE.EXC'] = wrap_ufunc(
    functools.partial(xpercentile, exclusive=True), **_percentile_kw
)
FUNCTIONS['_XLFN.PERCENTILE.INC'] = FUNCTIONS['PERCENTILE.INC'] = wrap_ufunc(
    xpercentile, **_percentile_kw
)


def xquartile(v, q, exclusive=False):
    if len(v) == 0:
        return Error.errors['#NUM!']
    if exclusive:
        n = len(v)
        rank = (n + 1) * q * 0.25
        if q <= 0 or q >= 4 or rank < 1 or rank > n:
            return Error.errors['#NUM!']
        method = 'weibull'
    else:
        if q < 0 or q > 4:
            return Error.errors['#NUM!']
        method = 'linear'
    return np.quantile(v, q * 0.25, method=method)


_quartile_kw = sh.combine_dicts(_percentile_kw, {
    'excluded': {0},
    'input_parser': lambda v, q: (v, np.floor(_convert_args(q))),
    'check_error': lambda *a: get_error(*a[::-1]),
    'args_parser': lambda v, q: (
        list(flatten(v, drop_empty=True)), replace_empty(q)
    )
})
FUNCTIONS['_XLFN.QUARTILE.EXC'] = FUNCTIONS['QUARTILE.EXC'] = wrap_ufunc(
    functools.partial(xquartile, exclusive=True), **_quartile_kw
)
FUNCTIONS['_XLFN.QUARTILE.INC'] = FUNCTIONS['QUARTILE.INC'] = wrap_ufunc(
    xquartile, **_quartile_kw
)


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
