#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2021 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
Python equivalents of financial Excel functions.
"""
import numpy as np
from . import (
    get_error, Error, wrap_func, raise_errors, text2num, flatten, Array,
    replace_empty, _text2num
)

FUNCTIONS = {}


def _xnpv(values, dates=None, min_date=0):
    err = get_error(dates, values)
    if not err and \
            any(isinstance(v, bool) for v in flatten((dates, values), None)):
        err = Error.errors['#VALUE!']
    if err:
        return lambda rate: err, None

    values, dates = tuple(map(replace_empty, (values, dates)))
    _ = lambda x: np.array(text2num(replace_empty(x)), float).ravel()
    if dates is None:
        values = _(values)
        t = np.arange(1, values.shape[0] + 1)
    else:
        dates = np.floor(_(dates))
        i = np.argsort(dates)
        values, dates = _(values)[i], dates[i]
        if len(values) != len(dates) or (dates <= min_date).any() or \
                (dates >= 2958466).any():
            return lambda rate: Error.errors['#NUM!'], None
        t = (dates - dates[0]) / 365

    def func(rate):
        return (values / np.power(1 + rate, t)).sum()

    t1, tv = t + 1, -t * values

    def dfunc(rate):
        return (tv / np.power(1 + rate, t1)).sum()

    return func, dfunc


def xnpv(rate, values, dates=None):
    with np.errstate(divide='ignore', invalid='ignore'):
        func = _xnpv(values, dates)[0]

        def _(r):
            e = isinstance(r, str) and Error.errors['#VALUE!']
            return get_error(r, e) or func(r)

        rate = text2num(replace_empty(rate))
        return np.vectorize(_, otypes=[object])(rate).view(Array)


def xxnpv(rate, values, dates):
    rate = np.asarray(rate)
    if rate.size > 1:
        return Error.errors['#VALUE!']
    raise_errors(rate)
    rate = _text2num(replace_empty(rate).ravel()[0])
    if isinstance(rate, (bool, str)):
        return Error.errors['#VALUE!']
    if rate <= 0:
        return Error.errors['#NUM!']

    return xnpv(rate, values, dates)


FUNCTIONS['NPV'] = wrap_func(lambda r, v, *a: xnpv(r, tuple(flatten((v, a)))))
FUNCTIONS['XNPV'] = wrap_func(xxnpv)


def xirr(values, guess=0.1):
    with np.errstate(divide='ignore', invalid='ignore'):
        import numpy_financial as npf
        res = npf.irr(tuple(flatten(text2num(replace_empty(values)).ravel())))
        res = (not np.isfinite(res)) and Error.errors['#NUM!'] or res

        def _(g):
            e = isinstance(g, str) and Error.errors['#VALUE!']
            return get_error(g, e) or res

        guess = text2num(replace_empty(guess))
        return np.vectorize(_, otypes=[object])(guess).view(Array)


FUNCTIONS['IRR'] = wrap_func(xirr)


def _newton(f, df, x, tol=.0000001):
    xmin = tol - 1
    with np.errstate(divide='ignore', invalid='ignore'):
        for _ in range(100):
            dx = f(x) / df(x)
            if not np.isfinite(dx):
                break
            if abs(dx) <= tol:
                return x
            x = max(xmin, x - dx)
    return Error.errors['#NUM!']


def xxirr(values, dates, x=0.1):
    x = np.asarray(x, object)
    if x.size > 1:
        return Error.errors['#VALUE!']
    raise_errors(x)
    x = _text2num(replace_empty(x).ravel()[0])
    if isinstance(x, (bool, str)):
        return Error.errors['#VALUE!']
    if x < 0:
        return Error.errors['#NUM!']
    f, df = _xnpv(values, dates, min_date=-1)
    if df is None:
        return f(x)
    return _newton(f, df, x)


FUNCTIONS['XIRR'] = wrap_func(xxirr)
