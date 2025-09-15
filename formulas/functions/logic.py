#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2025 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
Python equivalents of logical Excel functions.
"""
import functools
import numpy as np
from . import (
    wrap_ufunc, Error, flatten, get_error, wrap_func, XlError, raise_errors,
    Array, replace_empty, _convert2float, return_2d_func
)

FUNCTIONS = {}


def xif(condition, x, y=False):
    if isinstance(condition, str):
        return Error.errors['#VALUE!']
    return x if condition else y


def solve_cycle(*args):
    return not args[0]


FUNCTIONS['IF'] = {
    'function': wrap_ufunc(
        xif, input_parser=lambda *a: a,
        check_error=lambda cond, *a: get_error(cond)
    ),
    'solve_cycle': solve_cycle
}


def xifs(*cond_vals):
    if len(cond_vals) % 2:
        cond_vals += 0,
    for b, v in zip(cond_vals[::2], cond_vals[1::2]):
        err = get_error(b)
        if err:
            return err
        if isinstance(b, str):
            raise ValueError
        if b:
            return v
    return Error.errors['#N/A']


FUNCTIONS['_XLFN.IFS'] = FUNCTIONS['IFS'] = {
    'function': wrap_ufunc(
        xifs, input_parser=lambda *a: a,
        check_error=lambda *a: None
    ),
    'solve_cycle': lambda *a: not any(a[::2])
}


def xiferror(val, val_if_error):
    from .info import iserror
    return val_if_error if iserror(val) else val


FUNCTIONS['IFERROR'] = {
    'function': wrap_ufunc(
        xiferror, input_parser=lambda *a: a, check_error=lambda *a: False
    ),
    'solve_cycle': solve_cycle
}


def xifna(val, val_if_error):
    from .info import isna
    return val_if_error if isna(val) else val


FUNCTIONS['_XLFN.IFNA'] = FUNCTIONS['IFNA'] = {
    'function': wrap_ufunc(
        xifna, input_parser=lambda *a: a, check_error=lambda *a: False
    ),
    'solve_cycle': solve_cycle
}


def xswitch(val, *args):
    if isinstance(val, bool):
        condition = lambda x: val is x
    else:
        condition = lambda x: val == x
    for k, v in zip(args[::2], args[1::2]):
        if isinstance(k, XlError):
            return k
        elif condition(k):
            return v
    else:
        return args[-1] if len(args) % 2 else Error.errors['#N/A']


FUNCTIONS["_XLFN.SWITCH"] = FUNCTIONS["SWITCH"] = {
    'function': wrap_ufunc(
        xswitch, input_parser=lambda *a: a,
        check_error=lambda first, *a: get_error(first),
    )
}


def xand(logical, *logicals, func=np.logical_and.reduce):
    args = (logical,) + logicals
    raise_errors(args)
    check = lambda x: not isinstance(x, str)
    inp = tuple(flatten(args, check=check, drop_empty=True))
    return func(inp) if inp else Error.errors['#VALUE!']


FUNCTIONS['AND'] = {'function': wrap_func(xand)}
FUNCTIONS['OR'] = {'function': wrap_func(
    functools.partial(xand, func=np.logical_or.reduce)
)}
FUNCTIONS['_XLFN.XOR'] = FUNCTIONS['XOR'] = {'function': wrap_func(
    functools.partial(xand, func=np.logical_xor.reduce)
)}

FUNCTIONS['NOT'] = {'function': wrap_ufunc(
    np.logical_not, input_parser=lambda *a: a,
)}


def _get_first(v):
    if isinstance(v, np.ndarray):
        return v.ravel()[0]
    return v


def xbycol(array, func, axis=0):
    array = np.atleast_2d(array)
    if axis == 0:
        array = array.T
    res = np.asarray([[_get_first(func(v)) for v in array]], object)
    if axis == 1:
        res = res.T
    return res.view(Array)


FUNCTIONS['_XLFN.BYCOL'] = FUNCTIONS['BYCOL'] = wrap_func(
    functools.partial(xbycol, axis=0)
)
FUNCTIONS['_XLFN.BYROW'] = FUNCTIONS['BYROW'] = wrap_func(
    functools.partial(xbycol, axis=1)
)


def xmakearray(rows, cols, func):
    if rows.is_integer() and cols.is_integer() and rows >= 1 and cols >= 1:
        return np.asarray([[
            _get_first(func(r, c)) for c in np.arange(cols) + 1
        ] for r in np.arange(rows) + 1], object).tolist()
    return [[Error.errors['#VALUE!']]]


FUNCTIONS['_XLFN.MAKEARRAY'] = FUNCTIONS['MAKEARRAY'] = wrap_ufunc(
    xmakearray, input_parser=lambda rows, columns, func: (
        float(_convert2float(rows)), float(_convert2float(columns)), func
    ),
    check_error=lambda rows, columns, func: get_error(rows, columns, func),
    args_parser=lambda rows, columns, func: (
        replace_empty(rows), replace_empty(columns), func
    ), return_func=return_2d_func, check_nan=False, excluded={2}
)

FUNCTIONS['_XLFN.REDUCE'] = FUNCTIONS['REDUCE'] = wrap_ufunc(
    functools.reduce, input_parser=lambda initial_value, array, func: (
        func, array, initial_value
    ),
    check_error=lambda initial_value, array, func: get_error(func),
    args_parser=lambda initial_value, array, func: (
        replace_empty(initial_value), list(flatten(array, None)), func
    ), return_func=return_2d_func, check_nan=False, excluded={1, 2}
)


def xscan(initial_value, array, func):
    array = np.atleast_2d(array)
    out = []
    acc = replace_empty(initial_value)
    for x in flatten(array, None):
        acc = func(acc, x)
        out.append(_get_first(acc))
    return np.asarray(out).reshape(array.shape).view(Array)


FUNCTIONS['_XLFN.SCAN'] = FUNCTIONS['SCAN'] = wrap_func(xscan)


def xmap(array, func, *arrays):
    if arrays:
        arrays = (func,) + arrays
        func = arrays[-1]
        arrays = arrays[:-1]

    array = np.atleast_2d(array)
    shape = array.shape
    arrays = [array] + [np.atleast_2d(v) for v in arrays]
    if not all(shape == v.shape for v in arrays):
        return np.asarray([[Error.errors['#VALUE!']]]).view(Array)
    out = []
    for args in zip(*(flatten(v, None) for v in arrays)):
        out.append(_get_first(func(*args)))
    return np.asarray(out).reshape(array.shape).view(Array)


FUNCTIONS['_XLFN.MAP'] = FUNCTIONS['MAP'] = wrap_func(xmap)


def _true():
    return True


def _false():
    return False


FUNCTIONS['TRUE'] = wrap_func(_true)
FUNCTIONS['FALSE'] = wrap_func(_false)
