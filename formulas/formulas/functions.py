#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
Python equivalents of various excel functions.
"""

import collections
import math
import numpy as np
from ..errors import FunctionError


def is_number(number):
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


def xsum(*args):
    return sum(list(flatten(args)))


def average(*args):
    l = list(flatten(args))
    return sum(l) / len(l)


# noinspection PyUnusedLocal
def not_implemented(*args, **kwargs):
    raise FunctionError()


class Array(np.ndarray):
    pass


FUNCTIONS = collections.defaultdict(lambda: not_implemented)
FUNCTIONS.update({
    'INT': int,
    'PI': lambda: math.pi,
    'SUM': xsum,
    'AVERAGE': average,
    'ARRAYROW': lambda *args: np.asarray(args, object).view(Array),
    'ARRAY': lambda *args: np.asarray(args, object).view(Array),
    'AND': lambda *args: all(args),
    'OR': lambda *args: any(args),
})
