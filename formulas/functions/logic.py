#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2021 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
Python equivalents of logical Excel functions.
"""
import functools
import numpy as np
from . import (
    wrap_ufunc, Error, flatten, get_error, value_return, wrap_func, raise_errors
)

FUNCTIONS = {}


def xif(condition, x=True, y=False):
    if isinstance(condition, str):
        return Error.errors['#VALUE!']
    return x if condition else y


def solve_cycle(*args):
    return not args[0]


FUNCTIONS['IF'] = {
    'function': wrap_ufunc(
        xif, input_parser=lambda *a: a, return_func=value_return,
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
        xifs, input_parser=lambda *a: a, return_func=value_return,
        check_error=lambda *a: None
    ),
    'solve_cycle': lambda *a: not any(a[::2])
}


def xiferror(val, val_if_error):
    from .info import iserror
    return val_if_error if iserror(val) else val


# noinspection PyUnusedLocal
def xiferror_return(res, val, val_if_error):
    res._collapse_value = list(flatten(val_if_error, None))[0]
    return res


FUNCTIONS['IFERROR'] = {
    'function': wrap_ufunc(
        xiferror, input_parser=lambda *a: a, check_error=lambda *a: False,
        return_func=xiferror_return
    ),
    'solve_cycle': solve_cycle
}


def xswitch(val, *args):
    if isinstance(val, bool):
        condition = lambda x: val is x
    else:
        condition = lambda x: val == x
    for k, v in zip(args[::2], args[1::2]):
        if isinstance(k, Error):
            return k
        elif condition(k):
            return v
    else:
        return args[-1] if len(args) % 2 else Error.errors['#N/A']


FUNCTIONS["_XLFN.SWITCH"] = FUNCTIONS["SWITCH"] = {
    'function': wrap_ufunc(
        xswitch, input_parser=lambda *a: a, return_func=value_return,
        check_error=lambda first, *a: get_error(first),
    )
}


def xand(logical, *logicals, func=np.logical_and.reduce):
    check, arr = lambda x: not raise_errors(x) and not isinstance(x, str), []
    for a in (logical,) + logicals:
        v = list(flatten(a, check=check))
        arr.extend(v)
        if not v and not isinstance(a, np.ndarray):
            return Error.errors['#VALUE!']
    return func(arr) if arr else Error.errors['#VALUE!']


FUNCTIONS['AND'] = {'function': wrap_func(xand)}
FUNCTIONS['OR'] = {'function': wrap_func(
    functools.partial(xand, func=np.logical_or.reduce)
)}
FUNCTIONS['_XLFN.XOR'] = FUNCTIONS['XOR'] = {'function': wrap_func(
    functools.partial(xand, func=np.logical_xor.reduce)
)}

FUNCTIONS['NOT'] = {'function': wrap_ufunc(
    np.logical_not, input_parser=lambda *a: a, return_func=value_return
)}


def _true():
    return True


def _false():
    return False


FUNCTIONS['TRUE'] = wrap_func(_true)
FUNCTIONS['FALSE'] = wrap_func(_false)
