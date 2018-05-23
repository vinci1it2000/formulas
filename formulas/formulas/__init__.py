#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
It provides functions implementations to compile the excel formulas.

Sub-Modules:

.. currentmodule:: formulas.formulas

.. autosummary::
    :nosignatures:
    :toctree: formulas/

    ~functions
    ~operators
"""
import functools
import numpy as np
import schedula as sh
from ..errors import RangeValueError, FunctionError, FoundError, BaseError
from ..tokens.operand import Error


class Array(np.ndarray):
    _default = Error.errors['#N/A']

    def reshape(self, shape, *shapes, order='C'):
        try:
            return super(Array, self).reshape(shape, *shapes, order=order)
        except ValueError:
            res, (r, c) = np.empty(shape, object), self.shape
            res[:, :] = self._default
            r = None if r == 1 else r
            c = None if c == 1 else c
            try:
                res[:r, :c] = self
            except ValueError:
                res[:, :] = self.collapse(shape)
            return res

    def collapse(self, shape):
        return np.resize(self, shape)


# noinspection PyUnusedLocal
def not_implemented(*args, **kwargs):
    raise FunctionError()


def replace_empty(x, empty=0):
    if isinstance(x, np.ndarray):
        y = x.ravel().tolist()
        if sh.EMPTY in y:
            y = [empty if v is sh.EMPTY else v for v in y]
            return np.asarray(y, object).reshape(*x.shape)
    return x


# noinspection PyUnusedLocal
def wrap_func(func, ranges=False, **kw):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FoundError as ex:
            return np.asarray([[ex.err]], object)
        except BaseError as ex:
            raise ex
        except (ValueError, TypeError):
            return np.asarray([[Error.errors['#VALUE!']]], object)

    if not ranges:
        return wrap_ranges_func(functools.update_wrapper(wrapper, func))
    return functools.update_wrapper(wrapper, func)


def wrap_ranges_func(func, n_out=1):
    def wrapper(*args, **kwargs):
        try:
            args, kwargs = parse_ranges(*args, **kwargs)
            return func(*args, **kwargs)
        except RangeValueError:
            return sh.bypass(*((sh.NONE,) * n_out))

    return functools.update_wrapper(wrapper, func)


def parse_ranges(*args, **kw):
    from ..ranges import Ranges
    args = tuple(v.value if isinstance(v, Ranges) else v for v in args)
    kw = {k: v.value if isinstance(v, Ranges) else v for k, v in kw.items()}
    return args, kw
