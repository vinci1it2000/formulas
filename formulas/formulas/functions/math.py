#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
Python equivalents of math and trigonometry excel functions.
"""
import math
import functools
import numpy as np
from . import (
    get_error, raise_errors, is_number, flatten, wrap_ufunc, wrap_func,
    replace_empty, Error
)

# noinspection PyDictCreation
FUNCTIONS = {}
FUNCTIONS['ABS'] = wrap_ufunc(np.abs)
FUNCTIONS['ACOS'] = wrap_ufunc(np.arccos)
FUNCTIONS['ACOSH'] = wrap_ufunc(np.arccosh)
FUNCTIONS['_XLFN.ACOT'] = FUNCTIONS['ACOT'] = wrap_ufunc(
    lambda x: (np.arctan(np.divide(1, x)) + np.pi) % np.pi
)
FUNCTIONS['ACOTH'] = wrap_ufunc(lambda x: np.arctanh(np.divide(1, x)))
FUNCTIONS['_XLFN.ACOTH'] = FUNCTIONS['ACOTH']


def xarabic(text):
    num = (1000, 500, 100, 50, 10, 5, 1)
    it = map(num.__getitem__, map('MDCLXVI'.index, str(text).upper()[::-1]))
    res, add, p = 0, True, -1
    for v in it:
        if p != v:
            add = p < v
        p = v
        if add:
            res += v
        else:
            res -= v
    return res


FUNCTIONS['_XLFN.ARABIC'] = FUNCTIONS['ARABIC'] = wrap_ufunc(
    xarabic, input_parser=lambda *a: a,
    args_parser=lambda *a: (replace_empty(x, '') for x in a)
)

FUNCTIONS['ASIN'] = wrap_ufunc(np.arcsin)
FUNCTIONS['ASINH'] = wrap_ufunc(np.arcsinh)
FUNCTIONS['ATAN'] = wrap_ufunc(np.arctan)


def xarctan2(x, y):
    return x == y == 0 and Error.errors['#DIV/0!'] or np.arctan2(y, x)


FUNCTIONS['ATAN2'] = wrap_ufunc(xarctan2)
FUNCTIONS['ATANH'] = wrap_ufunc(np.arctanh)
FUNCTIONS['COS'] = wrap_ufunc(np.cos)
FUNCTIONS['COSH'] = wrap_ufunc(np.cosh)


def xcot(x, func=np.tan):
    x = func(x)
    return (1 / x) if x else Error.errors['#DIV/0!']


FUNCTIONS['COT'] = FUNCTIONS['_XLFN.COT'] = wrap_ufunc(xcot)
FUNCTIONS['COTH'] = FUNCTIONS['_XLFN.COTH'] = wrap_ufunc(
    functools.partial(xcot, func=np.tanh)
)
FUNCTIONS['CSC'] = FUNCTIONS['_XLFN.CSC'] = wrap_ufunc(
    functools.partial(xcot, func=np.sin)
)
FUNCTIONS['CSCH'] = FUNCTIONS['_XLFN.CSCH'] = wrap_ufunc(
    functools.partial(xcot, func=np.sinh)
)


def xceiling(num, sig, ceil=math.ceil, dfl=0):
    if sig == 0:
        return dfl
    elif sig < 0 < num:
        return np.nan
    return ceil(num / sig) * sig


FUNCTIONS['CEILING'] = wrap_ufunc(xceiling)


def xceiling_math(num, sig=None, mode=0, ceil=math.ceil):
    if sig == 0:
        return 0
    elif sig is None:
        x, sig = abs(num), 1
    else:
        sig = abs(sig)
        x = num / sig
    if mode and num < 0:
        return -ceil(abs(x)) * sig
    return ceil(x) * sig


FUNCTIONS['CEILING.MATH'] = wrap_ufunc(xceiling_math)
FUNCTIONS['_XLFN.CEILING.MATH'] = FUNCTIONS['CEILING.MATH']
FUNCTIONS['CEILING.PRECISE'] = FUNCTIONS['CEILING.MATH']
FUNCTIONS['_XLFN.CEILING.PRECISE'] = FUNCTIONS['CEILING.PRECISE']
FUNCTIONS['DEGREES'] = wrap_ufunc(np.degrees)


def xdecimal(text, radix):
    text, radix = str(text), int(radix)
    try:
        return int(text, radix)
    except ValueError:
        return Error.errors['#NUM!']


FUNCTIONS['_XLFN.DECIMAL'] = FUNCTIONS['DECIMAL'] = wrap_ufunc(
    xdecimal, input_parser=lambda *a: a
)


def xeven(x):
    v = math.ceil(abs(x) / 2.) * 2
    return -v if x < 0 else v


FUNCTIONS['EVEN'] = wrap_ufunc(xeven)
FUNCTIONS['EXP'] = wrap_ufunc(np.exp)


def xfact(number, fact=math.factorial, limit=0):
    return np.nan if number < limit else int(fact(int(number or 0)))


FUNCTIONS['FACT'] = wrap_ufunc(xfact)


def _factdouble(x):
    return np.multiply.reduce(np.arange(max(x, 1), 0, -2))


def xfactdouble(number):
    raise_errors(number)
    x = list(flatten(replace_empty(number), None))[0]
    if isinstance(x, bool):
        return Error.errors['#VALUE!']
    with np.errstate(divide='ignore', invalid='ignore'):
        # noinspection PyTypeChecker
        x = xfact(x, _factdouble, -1)
    return (np.isnan(x) or np.isinf(x)) and Error.errors['#NUM!'] or x


FUNCTIONS['FACTDOUBLE'] = wrap_func(xfactdouble)
FUNCTIONS['FLOOR'] = wrap_ufunc(
    functools.partial(xceiling, ceil=math.floor, dfl=Error.errors['#DIV/0!'])
)
FUNCTIONS['_XLFN.FLOOR.MATH'] = FUNCTIONS['FLOOR.MATH'] = wrap_ufunc(
    functools.partial(xceiling_math, ceil=math.floor)
)
FUNCTIONS['FLOOR.PRECISE'] = FUNCTIONS['FLOOR.MATH']
FUNCTIONS['_XLFN.FLOOR.PRECISE'] = FUNCTIONS['FLOOR.MATH']
FUNCTIONS['INT'] = wrap_ufunc(int)
FUNCTIONS['ISO.CEILING'] = FUNCTIONS['CEILING.PRECISE']
FUNCTIONS['LOG10'] = wrap_ufunc(np.log10)
FUNCTIONS['LOG'] = wrap_ufunc(
    lambda x, base=10: np.log(x) / np.log(base) if base else np.nan
)
FUNCTIONS['LN'] = wrap_ufunc(np.log)


def xmod(x, y):
    return y == 0 and Error.errors['#DIV/0!'] or np.mod(x, y)


FUNCTIONS['MOD'] = wrap_ufunc(xmod)


def xmround(*args):
    raise_errors(args)
    num, sig = list(flatten(map(replace_empty, args), None))
    if isinstance(num, bool) or isinstance(sig, bool):
        return Error.errors['#VALUE!']
    with np.errstate(divide='ignore', invalid='ignore'):
        x = num < 0 < sig and np.nan or xceiling(num, sig, ceil=np.round)
    return (np.isnan(x) or np.isinf(x)) and Error.errors['#NUM!'] or x


FUNCTIONS['MROUND'] = wrap_func(xmround)


def xodd(x):
    v = math.ceil(abs(x)) // 2 * 2 + 1
    return -v if x < 0 else v


FUNCTIONS['ODD'] = wrap_ufunc(xodd)
FUNCTIONS['PI'] = lambda: math.pi


def xpower(number, power):
    if number == 0:
        if power == 0:
            return Error.errors['#NUM!']
        if power < 0:
            return Error.errors['#DIV/0!']
    return np.power(number, power)


FUNCTIONS['POWER'] = wrap_ufunc(xpower)
FUNCTIONS['RADIANS'] = wrap_ufunc(np.radians)
FUNCTIONS['RAND'] = wrap_func(np.random.rand)


def xrandbetween(bottom, top):
    if isinstance(bottom, bool) or isinstance(top, bool):
        return Error.errors['#VALUE!']
    dx = top - bottom
    if dx < 0:
        return Error.errors['#NUM!']

    return bottom + dx * np.random.rand()


FUNCTIONS['RANDBETWEEN'] = wrap_ufunc(
    xrandbetween, input_parser=lambda *a: a,
    check_error=lambda *a: get_error(*a[::-1])
)


def _xroman(form):
    form = int(form + 1)
    num, let = (1000, 500, 100, 50, 10, 5, 1), 'MDCLXVI'
    for i, (n, l) in enumerate(zip(num, let), 1):
        yield n, l
        y = []
        for v, k in zip(num[i:], let[i:]):
            v = n - v
            if v not in num:
                y.append((v, k + l))
                if len(y) == form:
                    break
        yield from y[::-1]


def xroman(num, form=0):
    num, form = int(num), not isinstance(form, bool) and int(form or 0) or 0
    if not (0 <= num < 4000 and 0 <= form <= 4):
        raise ValueError()

    result = ""
    for i, n in _xroman(form):
        if not num:
            break
        count = int(num / i)
        result += n * count
        num -= i * count
    return result


FUNCTIONS['ROMAN'] = wrap_ufunc(xroman, input_parser=lambda *a: a)


def xround(x, d, func=round):
    d = 10 ** int(d)
    v = func(abs(x * d)) / d
    return -v if x < 0 else v


FUNCTIONS['ROUND'] = wrap_ufunc(xround)
FUNCTIONS['ROUNDDOWN'] = wrap_ufunc(functools.partial(xround, func=math.floor))
FUNCTIONS['ROUNDUP'] = wrap_ufunc(functools.partial(xround, func=math.ceil))
FUNCTIONS['SEC'] = FUNCTIONS['_XLFN.SEC'] = wrap_ufunc(
    functools.partial(xcot, func=np.cos)
)
FUNCTIONS['SECH'] = FUNCTIONS['_XLFN.SECH'] = wrap_ufunc(
    functools.partial(xcot, func=np.cosh)
)
FUNCTIONS['SIGN'] = wrap_ufunc(np.sign)
FUNCTIONS['SIN'] = wrap_ufunc(np.sin)
FUNCTIONS['SINH'] = wrap_ufunc(np.sinh)


def xsumproduct(*args):
    # Check all arrays are the same length
    # Excel returns #VAlUE! error if they don't match
    raise_errors(args)
    assert len(set(arg.size for arg in args)) == 1
    inputs = []
    for a in args:
        a = a.ravel()
        x = np.zeros_like(a, float)
        b = np.vectorize(is_number)(a)
        x[b] = a[b]
        inputs.append(x)

    return np.sum(np.prod(inputs, axis=0))


FUNCTIONS['SUMPRODUCT'] = wrap_func(xsumproduct)
FUNCTIONS['SQRT'] = wrap_ufunc(np.sqrt)


def xsrqtpi(number):
    raise_errors(number)
    x = list(flatten(replace_empty(number), None))[0]
    if isinstance(x, bool):
        return Error.errors['#VALUE!']
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.sqrt(float(x) * np.pi)
    return (np.isnan(x) or np.isinf(x)) and Error.errors['#NUM!'] or x


FUNCTIONS['SQRTPI'] = wrap_func(xsrqtpi)


def xsum(*args):
    raise_errors(args)
    return sum(list(flatten(args)))


FUNCTIONS['SUM'] = wrap_func(xsum)
FUNCTIONS['TAN'] = wrap_ufunc(np.tan)
FUNCTIONS['TANH'] = wrap_ufunc(np.tanh)
FUNCTIONS['TRUNC'] = wrap_ufunc(functools.partial(xround, func=math.trunc))
