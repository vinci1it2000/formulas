#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
Python equivalents of statistical Excel functions.
"""
import functools
import schedula as sh
from . import raise_errors, flatten, wrap_func, Error, is_number

FUNCTIONS = {}


def is_not_empty(v):
    return v is not sh.EMPTY


def _convert(v):
    if isinstance(v, str):
        return 0
    if isinstance(v, bool):
        return int(v)
    return v


def xfunc(*args, func=max, check=is_number, convert=None, default=0):
    raise_errors(args)
    it = flatten(args, check=check)
    return func(list(map(convert, it) if convert else it) or [default])


def xaverage(v):
    if v[0] is not None:
        return sum(v) / len(v)
    return Error.errors['#DIV/0!']


FUNCTIONS['AVERAGE'] = wrap_func(functools.partial(
    xfunc, func=xaverage, default=None
))
FUNCTIONS['AVERAGEA'] = wrap_func(functools.partial(
    xfunc, convert=_convert, check=is_not_empty, func=xaverage, default=None
))
FUNCTIONS['MAX'] = wrap_func(xfunc)
FUNCTIONS['MAXA'] = wrap_func(functools.partial(
    xfunc, convert=_convert, check=is_not_empty
))
FUNCTIONS['MIN'] = wrap_func(functools.partial(xfunc, func=min))
FUNCTIONS['MINA'] = wrap_func(functools.partial(
    xfunc, convert=_convert, check=is_not_empty, func=min
))
