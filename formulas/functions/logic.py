#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
Python equivalents of logical Excel functions.
"""
from . import wrap_ufunc, Array, Error, flatten, wrap_func, raise_errors, get_error

FUNCTIONS = {}


class IfArray(Array):
    def collapse(self, shape):
        if tuple(shape) == (1, 1) != self.shape:
            return Error.errors['#VALUE!']
        return super(IfArray, self).collapse(shape)


def xif(condition, x=True, y=False):
    if isinstance(condition, str):
        return Error.errors['#VALUE!']
    return x if condition else y


def solve_cycle(*args):
    return not args[0]


FUNCTIONS['IF'] = {
    'function': wrap_ufunc(
        xif, input_parser=lambda *a: a, otype=lambda *a: IfArray,
        check_error=lambda cond, *a: get_error(cond)
    ),
    'solve_cycle': solve_cycle
}


class IfErrorArray(Array):
    _value = Error.errors['#VALUE!']

    def collapse(self, shape):
        if tuple(shape) == (1, 1) != self.shape:
            return self._value
        return super(IfErrorArray, self).collapse(shape)


def xiferror(val, val_if_error):
    from .info import iserror
    return val_if_error if iserror(val) else val


# noinspection PyUnusedLocal
def xiferror_otype(val, val_if_error):
    class _IfErrorArray(IfErrorArray):
        _value = list(flatten(val_if_error, None))[0]

    return _IfErrorArray


FUNCTIONS['IFERROR'] = {
    'function': wrap_ufunc(
        xiferror, input_parser=lambda *a: a,
        check_error=lambda *a: False, otype=xiferror_otype
    ),
    'solve_cycle': solve_cycle
}

def _or(*args):
    raise_errors(args)
    return any(flatten(args, check=lambda x: isinstance(x, bool)))

FUNCTIONS['OR'] = wrap_func(_or)

def _and(*args):
    return all(flatten(args, check=lambda x: isinstance(x, bool)))

FUNCTIONS['AND'] = wrap_func(_and)

