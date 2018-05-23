#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
Python equivalents of logical excel functions.
"""
from . import wrap_ufunc, Array, Error, flatten

FUNCTIONS = {}


class IfArray(Array):
    def collapse(self, shape):
        if tuple(shape) == (1, 1) != self.shape:
            return Error.errors['#VALUE!']
        return super(IfArray, self).collapse(shape)


def xif(condition, x=True, y=False):
    return x if condition else y


FUNCTIONS['IF'] = wrap_ufunc(xif, otype=lambda *a: IfArray)


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


FUNCTIONS['IFERROR'] = wrap_ufunc(
    xiferror, input_parser=lambda *a: a,
    check_error=lambda *a: False, otype=xiferror_otype
)
