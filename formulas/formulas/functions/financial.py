#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
Python equivalents of financial excel functions.
"""
import numpy as np
from . import wrap_ufunc, flatten, get_error, Error

FUNCTIONS = {}


# noinspection PyUnusedLocal
def xirr(x, guess=0.1):
    if get_error(x):
        return Error.errors['#VALUE!']
    return np.irr(list(flatten(x)))


FUNCTIONS['IRR'] = wrap_ufunc(
    xirr, input_parser=lambda x, g=0.1: (x, float(g)),
    check_error=lambda *a: get_error(a[1:]), excluded={0}
)
