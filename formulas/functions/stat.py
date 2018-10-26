#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
Python equivalents of statistical Excel functions.
"""
from . import raise_errors, flatten, wrap_func, Error, excel_filter
import schedula as sh
import warnings

FUNCTIONS = {}


def xaverage(*args):
    raise_errors(args)
    v = list(flatten(args))
    if v:
        return sum(v) / len(v)
    return Error.errors['#DIV/0!']


FUNCTIONS['AVERAGE'] = wrap_func(xaverage)


def xmax(*args):
    raise_errors(args)
    return max(list(flatten(args)) or [0])


FUNCTIONS['MAX'] = wrap_func(xmax)


def xmin(*args):
    raise_errors(args)
    return min(list(flatten(args)) or [0])


FUNCTIONS['MIN'] = wrap_func(xmin)

def counta(*args):
    raise_errors(args)
    count = len(list(flatten(args, check=lambda x: x and x is not sh.EMPTY)))
    return count

FUNCTIONS['COUNTA'] = wrap_func(counta)

def countif(range, condition):
    return excel_filter(lambda x, y: x+1, range, condition)

FUNCTIONS['COUNTIF'] = wrap_func(countif)

