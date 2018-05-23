#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
It provides functions implementations to compile the excel functions.

Sub-Modules:

.. currentmodule:: formulas.formulas.functions

.. autosummary::
    :nosignatures:
    :toctree: functions/

    ~financial
    ~info
    ~logic
    ~look
    ~math
    ~stat
    ~text
"""
import functools
import importlib
import collections
import numpy as np
from .. import replace_empty, not_implemented, Array, wrap_func
from ...errors import FoundError, BroadcastError
from ...tokens.operand import XlError, Error

SUBMODULES = [
    '.info', '.logic', '.math', '.stat', '.financial', '.text', '.look'
]
# noinspection PyDictCreation
FUNCTIONS = {}
FUNCTIONS['ARRAY'] = lambda *args: np.asarray(args, object).view(Array)
FUNCTIONS['ARRAYROW'] = lambda *args: np.asarray(args, object).view(Array)


def get_error(*vals):
    # noinspection PyTypeChecker
    for v in flatten(vals, None):
        if isinstance(v, XlError):
            return v


def raise_errors(*args):
    # noinspection PyTypeChecker
    v = get_error(*args)
    if v:
        raise FoundError(err=v)


def is_number(number):
    if isinstance(number, bool):
        return False
    elif not isinstance(number, Error):
        try:
            float(number)
        except (ValueError, TypeError):
            return False
    return True


def flatten(l, check=is_number):
    if isinstance(l, collections.Iterable) and not isinstance(l, str):
        for el in l:
            yield from flatten(el, check)
    elif not check or check(l):
        yield l


def wrap_ufunc(
        func, input_parser=lambda *a: map(float, a), check_error=get_error,
        args_parser=lambda *a: map(replace_empty, a), otype=lambda *a: Array,
        **kw):
    """Helps call a numpy universal function (ufunc)."""

    def safe_eval(*vals):
        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                r = check_error(*vals) or func(*input_parser(*vals))
            if not isinstance(r, (XlError, str)):
                r = (np.isnan(r) or np.isinf(r)) and Error.errors['#NUM!'] or r
        except (ValueError, TypeError):
            r = Error.errors['#VALUE!']
        return r

    # noinspection PyUnusedLocal
    def wrapper(*args, **kwargs):
        try:
            args = tuple(args_parser(*args))
            kw['otypes'] = kw.get('otypes', [object])
            res = np.vectorize(safe_eval, **kw)(*args)
            try:
                return res.view(otype(*args))
            except AttributeError:
                return np.asarray(res).view(otype(*args))
        except ValueError as ex:
            try:
                np.broadcast(*args)
            except ValueError:
                raise BroadcastError()
            raise ex

    return wrap_func(functools.update_wrapper(wrapper, func), **kw)


@functools.lru_cache()
def get_functions():
    functions = collections.defaultdict(lambda: not_implemented)
    for name in SUBMODULES:
        functions.update(importlib.import_module(name, __name__).FUNCTIONS)
    functions.update(FUNCTIONS)
    return functions
