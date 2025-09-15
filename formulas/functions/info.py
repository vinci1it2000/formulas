#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2025 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
Python equivalents of information Excel functions.
"""
import functools
import collections
import numpy as np
import schedula as sh
from . import (
    wrap_ranges_func, Error, Array, XlError, wrap_func, is_number, flatten,
    _text2num, wrap_ufunc, raise_errors, DSP
)
from ..ranges import Ranges, _intersect
from ..tokens.function import LambdaFunction
from ..cell import _shape, _get_indices_intersection

FUNCTIONS = {}


class FalseArray(Array):
    _default = False


class TrueArray(Array):
    _default = True


def iserr(val):
    try:
        b = np.asarray([isinstance(v, XlError) and v is not Error.errors['#N/A']
                        for v in val.ravel().tolist()], bool)
        b.resize(val.shape)
        return b.view(FalseArray)
    except AttributeError:  # val is not an array.
        return iserr(np.asarray([[val]], object))[0][0].view(FalseArray)


FUNCTIONS['ISERR'] = wrap_ranges_func(iserr)


def xerrortype(val):
    if isinstance(val, XlError):
        return next((
            i for i, k in enumerate(Error.errors.values()) if k == val
        )) + 1
    return Error.errors['#N/A']


FUNCTIONS['ERROR.TYPE'] = wrap_ufunc(
    xerrortype, input_parser=lambda *a: a, check_error=lambda x: None,
    args_parser=lambda *a: a, check_nan=False
)


def isref(val):
    return isinstance(val, Ranges)


FUNCTIONS['ISREF'] = wrap_func(isref, True)


def iserror(val, check=lambda x: isinstance(x, XlError), array=TrueArray):
    try:
        b = np.asarray([check(v) for v in val.ravel().tolist()], bool)
        b.resize(val.shape)
        return b.view(array)
    except AttributeError:  # val is not an array.
        return iserror(
            np.asarray([[val]], object), check, array
        )[0][0].view(array)


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
    iserror, check=lambda x: is_number(x, xl_return=False), array=FalseArray
))
FUNCTIONS['ISBLANK'] = wrap_ranges_func(functools.partial(
    iserror, check=lambda x: x is sh.EMPTY, array=FalseArray
))
FUNCTIONS['ISTEXT'] = wrap_ranges_func(functools.partial(
    iserror, check=lambda x: isinstance(x, str) and not isinstance(x, sh.Token),
    array=FalseArray
))
FUNCTIONS['ISNONTEXT'] = wrap_ranges_func(functools.partial(
    iserror, check=lambda x: not isinstance(x, str) or isinstance(x, sh.Token),
    array=TrueArray
))
FUNCTIONS['ISLOGICAL'] = wrap_ranges_func(functools.partial(
    iserror, check=lambda x: isinstance(x, bool), array=FalseArray
))
FUNCTIONS['ISNA'] = wrap_ranges_func(functools.partial(
    iserror, check=isna, array=TrueArray
))


def xn(val):
    val = np.asarray(val, object).ravel()[0]
    raise_errors(val)
    return np.asarray([[
        val if isinstance(val, (bool, int, float)) else 0
    ]], object).view(Array)


FUNCTIONS['N'] = wrap_func(xn)


def xna():
    return Error.errors['#N/A']


FUNCTIONS['NA'] = wrap_func(xna)


def xtype(val):
    val = np.atleast_2d(val)
    if val.shape != (1, 1):
        return 64
    val = val.item()
    if isinstance(val, LambdaFunction):
        return 128
    if isinstance(val, XlError):
        return 16
    if isinstance(val, bool):
        return 4
    if isinstance(val, str) and val is not sh.EMPTY:
        return 2
    return 1


FUNCTIONS['TYPE'] = wrap_func(xtype)


def xisformula(dsp=None, ref=None):
    rng = ref.ranges[0]
    pred = dsp.solution.workflow.pred
    if rng['r1'] == rng['r2'] and rng['c1'] == rng['c2']:
        return any('=' in i for i in pred.get(rng['name'], ()))
    try:
        rng_ass = dsp.get_node(f"={rng['name']}")[0]
    except ValueError:
        return any('=' in i for i in pred.get(rng['name'], ()))
    base = rng_ass.range.ranges[0]
    out = np.empty(_shape(**base), object)
    out[:] = False
    for k, ind in rng_ass.inputs.items():
        if k is sh.SELF:
            for n, v in ind.items():
                if n in pred:
                    if isinstance(v, dict):
                        v = _get_indices_intersection(base, v)
                    i, j = v
                    out[i, j] = any('=' in i for i in pred[n])
        else:
            r = Ranges().push(k)
            ist = _intersect(base, r.ranges[0])
            if ist:
                br, bc = _get_indices_intersection(base, ist)
                v = xisformula(dsp, r)
                if isinstance(v, np.ndarray):
                    rr, rc = _get_indices_intersection(r.ranges[0], ist)
                    v = v[rr, rc]
                out[br, bc] = v
    return out.view(Array)


FUNCTIONS['_XLFN.ISFORMULA'] = FUNCTIONS['ISFORMULA'] = {
    'extra_inputs': collections.OrderedDict([(DSP, sh.EMPTY)]),
    'function': wrap_func(xisformula, ranges=True)
}
