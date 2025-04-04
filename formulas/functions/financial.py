#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2025 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
Python equivalents of financial Excel functions.
"""
import functools
import numpy as np
from . import (
    get_error, Error, wrap_func, raise_errors, text2num, flatten, Array,
    replace_empty, _text2num, wrap_ufunc, convert2float, _get_single_args
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


def _npf(func, *args, freturn=lambda x: x):
    import numpy_financial as npf
    r = getattr(npf, func)(*args)
    return freturn(r if getattr(r, 'shape', True) else r.ravel()[0])


FUNCTIONS['FV'] = wrap_ufunc(
    functools.partial(_npf, 'fv'),
    check_error=lambda *args: None,
    input_parser=lambda rate, nper, pmt, pv=0, type=0: convert2float(
        rate, nper, pmt, pv, type
    )
)


def xcumipmt(rate, nper, pv, start_period, end_period, type):
    args = rate, nper, pv, start_period, end_period, type
    args = tuple(map(_text2num, _get_single_args(*map(replace_empty, args))))
    raise_errors(*args)
    if any(not isinstance(v, (float, int)) for v in args):
        return Error.errors['#VALUE!']
    rate, nper, pv, start_period, end_period, type = args
    if rate <= 0 or nper <= 0 or pv <= 0 or start_period < 1 or \
            end_period < 1 or start_period > end_period or not type in (0, 1) \
            or nper < start_period or end_period > nper:
        return Error.errors['#NUM!']
    import numpy_financial as npf
    per = list(range(int(start_period), int(end_period + 1)))
    res = npf.ipmt(rate, per, nper, pv, fv=0, when=type)
    return res[res < 0].sum()


FUNCTIONS['CUMIPMT'] = wrap_func(xcumipmt)

_kw = {'input_parser': convert2float}
FUNCTIONS['PV'] = wrap_ufunc(functools.partial(_npf, 'pv'), **_kw)
FUNCTIONS['IPMT'] = wrap_ufunc(functools.partial(
    _npf, 'ipmt', freturn=lambda x: x > 0 and Error.errors['#NUM!'] or x,
), **_kw)
FUNCTIONS['PMT'] = wrap_ufunc(functools.partial(_npf, 'pmt'), **_kw)


def xppmt(rate, per, nper, pv, fv=0, type=0):
    import numpy_financial as npf
    if per < 1 or per >= nper + 1:
        return Error.errors['#NUM!']
    return npf.ppmt(rate, per, nper, pv, fv=fv, when=type)


FUNCTIONS['PPMT'] = wrap_ufunc(xppmt, **_kw)


def xrate(nper, pmt, pv, fv=0, type=0, guess=0.1):
    with np.errstate(over='ignore'):
        import numpy_financial as npf
        return npf.rate(
            nper, pmt, pv, fv=fv, when=type, guess=guess, maxiter=1000
        )


FUNCTIONS['RATE'] = wrap_ufunc(xrate, **_kw)


def xnper(rate, pmt, pv, fv=0, type=0):
    import numpy_financial as npf
    r = npf.nper(rate, pmt, pv, fv=fv, when=type)
    if rate == 0:
        r = np.sign(npf.nper(0.00000001, pmt, pv, fv=fv, when=type)) * np.abs(r)
    return r


FUNCTIONS['NPER'] = wrap_ufunc(xnper, **_kw)


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
