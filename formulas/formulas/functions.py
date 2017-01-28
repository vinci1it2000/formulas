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

from decimal import Decimal, ROUND_HALF_UP
import collections
try:
    import builtins
except:
    import __builtins__ as builtins

import numpy as np
import math
from ..errors import FunctionError


def flatten(l):
    if isinstance(l, collections.Iterable) and not isinstance(l, str):
        for el in l:
            yield from flatten(el)
    else:
        yield l


def xsum(*args):
    data = list(flatten(args))

    # however, if no non numeric cells, return zero (is what excel does)
    if len(data) < 1:
        return 0
    else:
        return sum(data)


def average(*args):
    l = list(flatten(*args))
    return sum(l) / len(l)


def is_number(number):
    try:
        float(number)
    except ValueError:
        return False
    return True


# Excel reference: https://support.office.com/en-us/article/ROUND-function-c018c5d8-40fb-4053-90b1-b3e7f61a213c
def xround(number, num_digits=0):
    if not is_number(number):
        raise TypeError("%s is not a number" % str(number))
    if not is_number(num_digits):
        raise TypeError("%s is not a number" % str(num_digits))

    if num_digits >= 0:  # round to the right side of the point
        return float(
            Decimal(repr(number)).quantize(Decimal(repr(pow(10, -num_digits))),
                                           rounding=ROUND_HALF_UP))
        # see https://docs.python.org/2/library/functions.html#round
        # and https://gist.github.com/ejamesc/cedc886c5f36e2d075c5

    else:
        return round(number, num_digits)


def xif(c, t, f=''):
    try:
        return t if c else f
    except ValueError:
        return t if c.any() else f


def date(year, month, day):  # Excel reference: https://support.office.com/en-us/article/DATE-function-e36c0c8c-4104-49da-ab83-82328b832349
    from datetime import datetime
    if type(year) != int:
        raise TypeError("%s is not an integer" % str(year))

    if type(month) != int:
        raise TypeError("%s is not an integer" % str(month))

    if type(day) != int:
        raise TypeError("%s is not an integer" % str(day))

    if year < 0 or year > 9999:
        raise ValueError(
            "Year must be between 1 and 9999, instead %s" % str(year))

    if year < 1900:
        year = 1900 + year

    date_0 = datetime(1900, 1, 1)
    date = datetime(year, month, day)

    result = (datetime(year, month, day) - date_0).days + 2

    if result <= 0:
        raise ArithmeticError("Date result is negative")
    else:
        return result


def iserror(value):
    return value in ()


def not_implemeted(*args, **kwargs):
    raise FunctionError()


FUNCTIONS = collections.defaultdict(lambda: not_implemeted)
FUNCTIONS.update((k.upper(), v) for k, v in builtins.__dict__.items())
FUNCTIONS.update({
    'PI': lambda: math.pi,
    'IF': xif,
    'SUM': xsum,
    'ROUND': xround,
    'AVERAGE': average,
    'ARRAYROW': lambda *args: np.array(args),
    'ARRAY': lambda *args: np.array(args),
    'DATE': date,
    'ISERROR': iserror,
    'AND': lambda *args: all(args),
    'OR': lambda *args: any(args),
})
