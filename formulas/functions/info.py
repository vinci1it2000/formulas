#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2021 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
Python equivalents of information Excel functions.
"""
import numpy as np
from . import wrap_ranges_func, Error, Array, XlError, wrap_func

FUNCTIONS = {}


class IsErrArray(Array):
    _default = False
    _collapse_value = True


def iserr(val):
    try:
        b = np.asarray([isinstance(v, XlError) and v is not Error.errors['#N/A']
                        for v in val.ravel().tolist()], bool)
        b.resize(val.shape)
        return b.view(IsErrArray)
    except AttributeError:  # val is not an array.
        return iserr(np.asarray([[val]], object))[0][0].view(IsErrArray)


FUNCTIONS['ISERR'] = wrap_ranges_func(iserr)


class IsErrorArray(IsErrArray):
    _default = True


def iserror(val):
    try:
        b = np.asarray([isinstance(v, XlError)
                        for v in val.ravel().tolist()], bool)
        b.resize(val.shape)
        return b.view(IsErrorArray)
    except AttributeError:  # val is not an array.
        return iserror(np.asarray([[val]], object))[0][0].view(IsErrorArray)


FUNCTIONS['ISERROR'] = wrap_ranges_func(iserror)


def xna():
    return Error.errors['#N/A']


FUNCTIONS['NA'] = wrap_func(xna)
